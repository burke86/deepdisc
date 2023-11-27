from deepdisc.data_format.file_io import get_data_from_json
from deepdisc.data_format.flatten import flatten_dc2
import os
import pytest


def test_flatten_shape(dc2_single_test_dict):
    ddicts = get_data_from_json(dc2_single_test_dict)
    flatdat = flatten_dc2(ddicts)
    
    assert len(flatdat)>0
    assert len(flatdat[0]) == 98316




