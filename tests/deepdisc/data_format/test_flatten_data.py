from deepdisc.data_format.file_io import get_data_from_json
from deepdisc.data_format.flatten import flatten_dc2
from deepdisc.data_format.image_readers import DC2ImageReader

import os
import pytest

def blank_key_mapper(dataset_dict):
    return dataset_dict['filename']

def test_flatten_shape(dc2_single_test_dict):
    image_reader = DC2ImageReader(norm="raw")
    ddicts = get_data_from_json(dc2_single_test_dict)
    flatdat = flatten_dc2(ddicts, image_reader, blank_key_mapper)
    
    assert len(flatdat)>0
    assert len(flatdat[0]) == 98317




