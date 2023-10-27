import os
import pytest

from deepdisc.data_format.file_io import get_data_from_json


def test_get_data_from_json(tmp_path):
    """Write a little test file and make sure we can read it back."""
    test_file = os.path.join(tmp_path, "test.json")
    with open(test_file, "w", encoding="utf-8") as file_handle:
        file_handle.write('{"string":"the quick brown fox", "integer": 9}')
    data_dict = get_data_from_json(test_file)

    assert len(data_dict) == 2
    expected_dict = {"string": "the quick brown fox", "integer": 9}
    assert data_dict == expected_dict


def test_get_data_from_json_hsc(hsc_single_test_file):
    """Test that we can parse the test HSC data."""
    data_dict = get_data_from_json(filename=hsc_single_test_file)
    assert len(data_dict) > 0

    error_filename = "./file_does_not_exist.json"
    with pytest.raises(Exception):
        _ = get_data_from_json(error_filename)
