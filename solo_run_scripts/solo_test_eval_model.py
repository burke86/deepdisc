"""
This code will read in a trained model and output the classes for predicted objects matched to the ground truth 

"""
import logging
import os
import time

import numpy as np
import deepdisc.astrodet.astrodet as toolkit

from deepdisc.data_format.file_io import get_data_from_json
from deepdisc.data_format.image_readers import HSCImageReader
from deepdisc.inference.match_objects import get_matched_object_classes
from deepdisc.inference.predictors import return_predictor_transformer
from deepdisc.utils.parse_arguments import dtype_from_args, make_inference_arg_parser

from detectron2 import model_zoo
from detectron2.config import get_cfg, LazyConfig
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

from pathlib import Path

setup_logger()
logger = logging.getLogger(__name__)

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

def return_predictor(
    cfgfile, run_name, nc=1, output_dir="/home/shared/hsc/HSC/HSC_DR3/models/noclass/", roi_thresh=0.5
):
    """
    This function returns a trained model and its config file.
    Used for models that have yacs config files

    Parameters
    ----------
    cfgfile: str
        A path to a model config file, provided by the detectron2 repo
    run_name: str
        Prefix used for the name of the saved model
    nc: int
        Number of classes used in the model
    output_dir: str
        THe directory to save metric outputs
    roi_thresh: float
        Hyperparamter that functions as a detection sensitivity level

    """

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfgfile))  # Get model structure
    cfg.DATASETS.TRAIN = "astro_train"  # Register Metadata
    cfg.DATASETS.TEST = ("astro_test",)  # Config calls this TEST, but it should be the val dataset
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = (
        4  # this is images per iteration. 1 epoch is len(images)/(ims_per_batch iterations*num_gpus)
    )
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.STEPS = []  # do not decay learning rate for retraining
    cfg.SOLVER.MAX_ITER = 100  # for DefaultTrainer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        512  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nc
    cfg.OUTPUT_DIR = output_dir
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.INPUT.MIN_SIZE_TRAIN = 1025
    cfg.INPUT.MAX_SIZE_TRAIN = 1050

    # Defaults
    # PRE_NMS_TOPK_TEST: 6000
    # POST_NMS_TOPK_TEST: 1000
    # PRE_NMS_TOPK_TRAIN: 12000
    # POST_NMS_TOPK_TRAIN: 2000

    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000

    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, run_name)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_thresh  # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.MODEL.ROI_BOX_HEAD.GAMMA = 1
    cfg.MODEL.ROI_BOX_HEAD.ALPHAS = None

    predictor = toolkit.AstroPredictor(cfg)

    return predictor, cfg


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

if __name__ == "__main__":
    # Handle args
    args = make_inference_arg_parser().parse_args()
    output_dir = args.output_dir
    roi_thresh = args.roi_thresh
    run_name = args.run_name
    testfile = args.testfile
    savedir = args.savedir
    Path(savedir).mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir
    roi_thresh=args.roi_thresh
    run_name=args.run_name
    dtype=dtype_from_args(args.datatype)

    # Load data
    dataset_names = ["test"]
    datadir = "/home/shared/hsc/HSC/HSC_DR3/data/"
    t0 = time.time()
    dataset_dicts = {}
    for i, d in enumerate(dataset_names):
        dataset_dicts[d] = get_data_from_json(testfile)
    print("Took ", time.time() - t0, "seconds to load samples")
    
    # Local vars/metadata
    classes = ["star", "galaxy"]
    bb = args.run_name.split("_")[0]
    
    # TODO throw a bunch of this into a main() fn to match test_run_transformers
    
    # --------- Setting a bunch of config stuff
    cfgfile = (
        f"./tests/deepdisc/test_data/configs/"
        f"solo/solo_cascade_mask_rcnn_swin_b_in21k_50ep_test_eval.py"
    )
    cfg = LazyConfig.load(cfgfile)

    cfg.model.roi_heads.num_classes = args.nc

    for bp in cfg.model.roi_heads.box_predictors:
        bp.test_score_thresh = roi_thresh

    for box_predictor in cfg.model.roi_heads.box_predictors:
        box_predictor.test_topk_per_image = 1000
        box_predictor.test_score_thresh = roi_thresh

    cfg_loader = get_cfg()

    cfg.train.init_checkpoint = os.path.join(output_dir, run_name)
    
    # --------- Now we case predictor on model type (the second case has way different config vals it appears)
    # predictor = return_predictor_transformer(cfg,cfg_loader)
    
    cfg.OUTPUT_DIR = output_dir
    if bb in ['Swin','MViTv2']:
        predictor= return_predictor_transformer(cfg,cfg)#cfg_loader)
    else:
        predictor, cfg = return_predictor(cfgfile, run_name, output_dir=output_dir, nc=2, roi_thresh=roi_thresh)

    # --------- This section is similar to test_run_transformers
    def hsc_key_mapper(dataset_dict):
        filenames = [
            dataset_dict["filename_G"],
            dataset_dict["filename_R"],
            dataset_dict["filename_I"],
        ]
        return filenames
    IR = HSCImageReader(norm=args.norm)
    
    # --------- Do the thing
    t0 = time.time()
    print("Matching objects")
    true_classes, pred_classes = get_matched_object_classes(dataset_dicts["test"], IR, hsc_key_mapper, predictor)
    classes = np.array([true_classes, pred_classes])

    savename = f"{bb}_test_matched_classes.npy"
    np.save(os.path.join(args.savedir, savename), classes)

    print("Took ", time.time() - t0, " seconds")
    print(classes)
    t0 = time.time()