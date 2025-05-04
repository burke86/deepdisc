import pytest

from deepdisc.data_format.register_data import register_data_set

def test_register_data(hsc_single_test_file):
    meta = register_data_set("astro_train", filename=hsc_single_test_file, thing_classes=["star", "galaxy"])
    assert meta is not None
    print(meta)

    # Test that loading an invalid file fails.
    error_filename = "./file_does_not_exist.json"
    with pytest.raises(Exception):
        meta = register_data_set("astro_train2", error_filename, thing_classes=["star", "galaxy"])
