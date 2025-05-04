from deepdisc.data_format.file_io import get_data_from_json
from deepdisc.data_format.image_readers import DC2ImageReader

import os
import pytest

def blank_key_mapper(dataset_dict):
    return dataset_dict['filename']





