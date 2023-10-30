# Training script for LazyConfig models
try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

except:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import copy
import gc
import logging
import os
import random
import sys
import time
import weakref
from typing import Dict, List, Optional

import cv2
import detectron2.checkpoint as checkpointer
import detectron2.data as data
import detectron2.data.transforms as T
import detectron2.modeling as modeler
import detectron2.solver as solver
import detectron2.utils.comm as comm
import imgaug.augmenters.blur as blur
import imgaug.augmenters.flip as flip

# import some common libraries
import numpy as np
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.engine import (
    DefaultPredictor,
    DefaultTrainer,
    HookBase,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.utils.visualizer import Visualizer

# from astrodet import astrodet as toolkit
# from astrodet import detectron as detectron_addons

import glob

from astropy.io import fits
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.solver import build_lr_scheduler
from detectron2.structures import BoxMode

from deepdisc.data_format.register_data import register_data_set
from deepdisc.model.loaders import return_test_loader, return_train_loader
from deepdisc.model.models import return_lazy_model
from deepdisc.training.trainers import (
    return_evallosshook,
    return_lazy_trainer,
    return_optimizer,
    return_savehook,
    return_schedulerhook,
)
from deepdisc.utils.parse_arguments import dtype_from_args, make_training_arg_parser


def main(train_head, args):
    # Hack if you get SSL certificate error
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    output_dir = args.output_dir
    output_name = args.run_name
    dirpath = args.data_dir  # Path to dataset
    scheme = args.scheme
    alphas = args.alphas
    modname = args.modname
    if modname == "swin":
        cfgfile = "./tests/deepdisc/test_data/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py"
        # initwfile = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_246a82.pkl"
    elif modname == "mvitv2":
        cfgfile = "/home/shared/hsc/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py"
        # initwfile = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_8c3da3.pkl"

    elif modname == "vitdet":
        cfgfile = "/home/shared/hsc/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
        # initwfile = '/home/g4merz/deblend/detectron2/projects/ViTDet/model_final_435fa9.pkl'
        # initwfile = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_61ccd1.pkl"

    dtype = dtype_from_args(args.dtype)
    trainfile = dirpath + "single_test.json"
    testfile = dirpath + "single_test.json"

    classes = ["star", "galaxy"]
    numclasses = len(classes)

    # Register the data sets and get the metadata.
    astrotrain_metadata = register_data_set("astro_train", trainfile, thing_classes=classes)
    astroval_metadata = register_data_set("astro_val", testfile, thing_classes=classes)

    cfg = LazyConfig.load(cfgfile)

    # metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names) # to get labels from ids

    bs = 1
    cfg.model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
    cfg.dataloader.train.total_batch_size = bs
    cfg.model.roi_heads.num_classes = numclasses
    cfg.model.roi_heads.batch_size_per_image = 512

    cfg_loader = get_cfg()
    cfg_loader.SOLVER.IMS_PER_BATCH = bs
    cfg_loader.DATASETS.TRAIN = "astro_train"  # Register Metadata
    cfg_loader.DATASETS.TEST = "astro_val"
    # cfg_loader.DATALOADER.NUM_WORKERS = 0
    cfg_loader.DATALOADER.PREFETCH_FACTOR = 2
    cfg_loader.SOLVER.BASE_LR = 0.001
    cfg_loader.OUTPUT_DIR = output_dir
    os.makedirs(cfg_loader.OUTPUT_DIR, exist_ok=True)

    cfg_loader.SOLVER.CLIP_GRADIENTS.ENABLED = True
    # Type of gradient clipping, currently 2 values are supported:
    # - "value": the absolute values of elements of each gradients are clipped
    # - "norm": the norm of the gradient for each parameter is clipped thus
    #   affecting all elements in the parameter
    cfg_loader.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    # Maximum absolute value used for clipping gradients
    # Floating point number p for L-p norm to be used with the "norm"
    # gradient clipping type; for L-inf, please specify .inf
    cfg_loader.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0

    # iterations for 15,25,35,50 epochs
    epoch = int(args.tl / cfg.dataloader.train.total_batch_size)
    e1 = 20
    e2 = epoch * 10
    e3 = epoch * 20
    efinal = epoch * 35

    val_per = 5

    if train_head:
        # cfg.train.init_checkpoint = initwfile # replace with the path were you have your model
        cfg.train.init_checkpoint = None

        # Step 1)

        model = return_lazy_model(cfg)

        cfg.optimizer.params.model = model
        cfg.optimizer.lr = 0.001

        cfg_loader.SOLVER.STEPS = []  # do not decay learning rate for retraining
        cfg_loader.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg_loader.SOLVER.WARMUP_ITERS = 0
        cfg_loader.SOLVER.MAX_ITER = e1  # for DefaultTrainer

        # optimizer = instantiate(cfg.optimizer)

        optimizer = return_optimizer(cfg)

        loader = return_train_loader(
            cfg_loader,
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
        )
        test_loader = return_test_loader(
            cfg_loader,
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
        )

        saveHook = return_savehook(output_name)
        lossHook = return_evallosshook(val_per, model, test_loader)
        schedulerHook = return_schedulerhook(optimizer)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = return_lazy_trainer(model, loader, optimizer, cfg, cfg_loader, hookList)

        trainer.set_period(5)
        trainer.train(0, 20)
        if comm.is_main_process():
            np.save(output_dir + output_name + "_losses", trainer.lossList)
            np.save(output_dir + output_name + "_val_losses", trainer.vallossList)
        return


if __name__ == "__main__":
    args = make_training_arg_parser().parse_args()
    print("Command Line Args:", args)

    print("Training head layers")
    train_head = True
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            train_head,
            args,
        ),
    )

    torch.cuda.empty_cache()
    gc.collect()

    print(f"Took {time.time()-t0} seconds")
