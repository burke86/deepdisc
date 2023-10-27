import pytest

from deepdisc.data_format.register_data import register_data_set

from utils_for_tests.data_paths import get_test_hsc_data_path


def test_register_data():
    filename = get_test_hsc_data_path("single_test.json")
    meta = register_data_set("astro_train", filename, thing_classes=["star", "galaxy"])
    assert meta is not None
    print(meta)

    # Test that loading an invalid file fails.
    error_filename = get_test_hsc_data_path("file_does_not_exist.json")
    with pytest.raises(Exception):
        meta = register_data_set("astro_train2", error_filename, thing_classes=["star", "galaxy"])
