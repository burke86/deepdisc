""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #

classes = ["star", "galaxy"]
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
model.roi_heads.num_classes = len(classes) # args.nc ##TODO assuming nc can be replaced with the number of classes we declared above
model.roi_heads.batch_size_per_image = 512


model.proposal_generator.pre_nms_topk = [6000, 6000]
model.proposal_generator.post_nms_topk = [6000, 6000]
model.proposal_generator.nms_thresh = 0.3

# ---------------------------------------------------------------------------- #
# Yaml-style config (was formerly saved as a .yaml file, loaded to cfg_loader)
# ---------------------------------------------------------------------------- #
# Get values from template
from .yacs_style_defaults import MISC, DATALOADER, DATASETS, GLOBAL, INPUT, MODEL, SOLVER, TEST

# Overrides
DATALOADER.NUM_WORKERS = 1

DATASETS.TRAIN = "astro_train"  # Register Metadata
DATASETS.TEST = "astro_val"

SOLVER.BASE_LR = 0.001
SOLVER.IMS_PER_BATCH = 4

TEST.DETECTIONS_PER_IMAGE = 1000