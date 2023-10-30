import os
import pytest

from deepdisc.data_format.image_readers import DC2ImageReader, HSCImageReader

def test_read_hsc_data(hsc_triple_test_file):
    """Test that we can read the test DC2 data."""
    ir = HSCImageReader(norm="raw")
    img = ir(hsc_triple_test_file)
    assert img.shape[0] == 1050
    assert img.shape[1] == 1025
    assert img.shape[2] == 3


def test_read_dc2_data(dc2_single_test_file):
    """Test that we can read the test DC2 data."""
    ir = DC2ImageReader(norm="raw")
    img = ir(dc2_single_test_file)
    assert img.shape[0] == 525
    assert img.shape[1] == 525
    assert img.shape[2] == 6
