""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #


metadata = OmegaConf.create() 
metadata.classes = ["star", "galaxy"]

output_dir = "/home/shared/hsc/HSC/HSC_DR3/models/noclass/"

nc=1 # Number of classes used in the model
####TODO: why is this 1 when we've defined two classes above?
roi_thresh = 0.5 # Hyperparamter that functions as a detection sensitivity level

#prev
#numclasses = len(metadata.classes)
#bs = 1

# ---------------------------------------------------------------------------- #
# Standard config (this has always been the LazyConfig/.py-style config)
# ---------------------------------------------------------------------------- #
# Get values from templates
from ..COCO.cascade_mask_rcnn_swin_b_in21k_50ep import dataloader, lr_multiplier, model, optimizer, train

# Overrides
dataloader.train.total_batch_size = 4
model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
#model.roi_heads.num_classes = args.nc ####TODO leaving out for now
model.roi_heads.batch_size_per_image = 512
#Prev:
# Overrides
#model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
#dataloader.train.total_batch_size = bs
#model.roi_heads.num_classes = numclasses
#model.roi_heads.batch_size_per_image = 512

# ---------------------------------------------------------------------------- #
# Yaml-style config (was formerly saved as a .yaml file, loaded to cfg_loader)
# ---------------------------------------------------------------------------- #
# Get values from template
from .yacs_style_defaults import MISC, DATALOADER, DATASETS, GLOBAL, INPUT, MODEL, SOLVER, TEST


# Overrides
DATALOADER.NUM_WORKERS = 1

DATASETS.TRAIN = "astro_train" # Register Metadata
DATASETS.TEST = ("astro_test",) # Config calls this TEST, but it should be the val dataset

INPUT.MIN_SIZE_TRAIN = 1025
INPUT.MAX_SIZE_TRAIN = 1050

MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
MODEL.RPN.POST_NMS_TOPK_TEST = 6000
MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
MODEL.ROI_BOX_HEAD.ALPHAS = None
MODEL.ROI_BOX_HEAD.GAMMA = 1
MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    512  # faster, and good enough for this toy dataset (default: 512)
)
MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
MODEL.ROI_HEADS.NUM_CLASSES = nc
MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_thresh # set a custom testing threshold
MODEL.WEIGHTS = os.path.join(output_dir, run_name)  # path to the model we just trained ####TODO

OUTPUT_DIR = output_dir ####TODO

SOLVER.IMS_PER_BATCH = (
    4  # this is images per iteration. 1 epoch is len(images)/(ims_per_batch iterations*num_gpus)
)
SOLVER.BASE_LR = 0.001
SOLVER.STEPS = []  # do not decay learning rate for retraining
SOLVER.MAX_ITER = 100  # for DefaultTrainer

TEST.DETECTIONS_PER_IMAGE = 1000





#Prev:
# Overrides
#DATALOADER.PREFETCH_FACTOR = 2

#DATASETS.TRAIN = "astro_train"
#DATASETS.TEST = "astro_val"

#SOLVER.BASE_LR = 0.001
#SOLVER.CLIP_GRADIENTS.ENABLED = True
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
#SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
#SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0
#SOLVER.IMS_PER_BATCH = bs
