""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
"""
bs = 1

metadata = OmegaConf.create() 
metadata.classes = ["star", "galaxy"]

numclasses = len(metadata.classes)
"""
# ---------------------------------------------------------------------------- #
# Standard config (this has always been the LazyConfig/.py-style config)
# ---------------------------------------------------------------------------- #
# Get values from templates
from ..COCO.cascade_mask_rcnn_swin_b_in21k_50ep import dataloader, lr_multiplier, model, optimizer, train

# Overrides
model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
dataloader.train.total_batch_size = 4
#model.roi_heads.num_classes = args.nc
model.roi_heads.batch_size_per_image = 512

#for bp in model.roi_heads.box_predictors:
#    bp.test_score_thresh = roi_thresh

model.proposal_generator.pre_nms_topk = [6000, 6000]
model.proposal_generator.post_nms_topk = [6000, 6000]
model.proposal_generator.nms_thresh = 0.3
"""
# Overrides
model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
dataloader.train.total_batch_size = bs
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512
"""

# ---------------------------------------------------------------------------- #
# Yaml-style config (was formerly saved as a .yaml file, loaded to cfg_loader)
# ---------------------------------------------------------------------------- #
# Get values from template
from .yacs_style_defaults import MISC, DATALOADER, DATASETS, GLOBAL, INPUT, MODEL, SOLVER, TEST

# Overrides
SOLVER.IMS_PER_BATCH = 4
DATASETS.TRAIN = "astro_train"  # Register Metadata
DATASETS.TEST = "astro_val"
DATALOADER.NUM_WORKERS = 1
SOLVER.BASE_LR = 0.001
#OUTPUT_DIR = output_dir

TEST.DETECTIONS_PER_IMAGE = 1000
"""
# Overrides
DATALOADER.PREFETCH_FACTOR = 2

DATASETS.TRAIN = "astro_train"
DATASETS.TEST = "astro_val"

SOLVER.BASE_LR = 0.001
SOLVER.CLIP_GRADIENTS.ENABLED = True
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0
SOLVER.IMS_PER_BATCH = bs
"""
