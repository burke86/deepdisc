import argparse
import numpy as np
import pytest

from deepdisc.utils.parse_arguments import (
    dtype_from_args,
    make_inference_arg_parser,
    make_training_arg_parser,
)


def test_make_inference_arg_parser():
    parser = make_inference_arg_parser()
    args = parser.parse_args()

    # Check all parameters are included and the default values are expected.
    assert args.datatype == 8
    assert args.nc == 2
    assert args.norm == "astrolupton"
    assert args.output_dir == "."
    assert args.roi_thresh == 0.1
    assert type(args.run_name) is str and len(args.run_name) > 0
    assert args.savedir == "."
    assert args.scheme == 2
    assert type(args.testfile) is str and len(args.testfile) > 0


def test_make_training_arg_parser():
    parser = make_training_arg_parser()
    args = parser.parse_args()

    # Check all parameters are included and the default values are expected.
    assert type(args.cfgfile) is str and len(args.cfgfile) > 0
    assert type(args.config_file) is str  # Default is empty
    assert type(args.data_dir) is str and len(args.data_dir) > 0
    assert type(args.output_dir) is str and len(args.output_dir) > 0
    assert type(args.run_name) is str and len(args.run_name) > 0

    assert not args.eval_only
    assert not args.from_scratch
    assert not args.resume

    assert args.num_gpus == 1
    assert args.num_machines == 1
    assert args.machine_rank == 0

    assert args.A == 1e3
    assert args.cp == 99.99
    assert not args.do_fl
    assert not args.do_norm
    assert args.dtype == 8
    assert args.modname == "swin"
    assert args.norm == "astrolupton"
    assert args.Q == 10
    assert args.scheme == 1
    assert args.stretch == 0.5
    assert args.tl == 1


def test_dtype_from_args():
    assert dtype_from_args(8) is np.uint8
    assert dtype_from_args(16) is np.int16

    with pytest.raises(ValueError):
        _ = dtype_from_args(4)
    with pytest.raises(ValueError):
        _ = dtype_from_args(32)  # Do we want to support this too?
