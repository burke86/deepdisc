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
from detectron2.utils.logger import setup_logger

setup_logger()

import gc
import os
import time

import detectron2.utils.comm as comm

# import some common libraries
import numpy as np
import torch

# import some common detectron2 utilities
from detectron2.config import LazyConfig, get_cfg
from detectron2.engine import launch

from deepdisc.data_format.image_readers import HSCImageReader
from deepdisc.data_format.register_data import register_data_set
from deepdisc.model.loaders import return_test_loader, return_train_loader, test_mapper_cls, train_mapper_cls
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
        cfgfile = "./tests/deepdisc/test_data/configs/solo/solo_cascade_mask_rcnn_swin_b_in21k_50ep.py"
        #cfgfile = "./tests/deepdisc/test_data/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py" #replaced
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

    #cfg_loader = get_cfg()
    #cfg.SOLVER.IMS_PER_BATCH = bs #replaced
    #cfg.DATASETS.TRAIN = "astro_train"  # Register Metadata #replaced
    #cfg.DATASETS.TEST = "astro_val" #replaced
    # cfg_loader.DATALOADER.NUM_WORKERS = 0 #replaced
    #cfg.DATALOADER.PREFETCH_FACTOR = 2 #replaced
    #cfg.SOLVER.BASE_LR = 0.001 #replaced
    
    cfg.OUTPUT_DIR = output_dir #replaced
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) #replaced

    #cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True #replaced
    # Type of gradient clipping, currently 2 values are supported:
    # - "value": the absolute values of elements of each gradients are clipped
    # - "norm": the norm of the gradient for each parameter is clipped thus
    #   affecting all elements in the parameter
    #cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm" #replaced
    # Maximum absolute value used for clipping gradients
    # Floating point number p for L-p norm to be used with the "norm"
    # gradient clipping type; for L-inf, please specify .inf
    #cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0 #replaced

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

        cfg.SOLVER.STEPS = []  # do not decay learning rate for retraining #replaced
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR" #replaced
        cfg.SOLVER.WARMUP_ITERS = 0 #replaced
        cfg.SOLVER.MAX_ITER = e1  # for DefaultTrainer #replaced

        # optimizer = instantiate(cfg.optimizer)

        optimizer = return_optimizer(cfg)

        # key_mapper function should take a dataset_dict as input and output a key used by the image_reader function
        def hsc_key_mapper(dataset_dict):
            filenames = [
                dataset_dict["filename_G"],
                dataset_dict["filename_R"],
                dataset_dict["filename_I"],
            ]
            return filenames

        IR = HSCImageReader(norm=args.norm)
        mapper = train_mapper_cls(IR, hsc_key_mapper)
        loader = return_train_loader(cfg, mapper) #replaced
        test_mapper = test_mapper_cls(IR, hsc_key_mapper)
        test_loader = return_test_loader(cfg, test_mapper) #replaced

        saveHook = return_savehook(output_name)
        lossHook = return_evallosshook(val_per, model, test_loader)
        schedulerHook = return_schedulerhook(optimizer)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = return_lazy_trainer(model, loader, optimizer, cfg, cfg, hookList) #replaced

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
    
