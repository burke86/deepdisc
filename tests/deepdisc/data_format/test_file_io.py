import os
import pytest

from deepdisc.data_format.file_io import get_data_from_json, DDLoader
from deepdisc.data_format.annotation_functions.annotate_hsc import annotate_hsc


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


def test_data_loader_generate_filedict():
    """Simple test to check generating file dict"""

    test_data_dirpath = 'tests/deepdisc/test_data/'

    # Initialize the DDLoader object
    hsc_loader = DDLoader()

    filters = ['G', 'R']

    # Generate the dictionary of file paths
    hsc_loader.generate_filedict(
        os.path.abspath(os.path.join(test_data_dirpath, 'hsc')),
        filters,
        '*_scarlet_img.fits',
        '*_scarlet_segmask.fits'
    )

    hsc_loader.filedict['filters']
    assert hsc_loader.filedict['filters'] == filters
    assert len(hsc_loader.filedict['G']['img']) == 1
    assert len(hsc_loader.filedict['R']['img']) == 1
    # assert len(hsc_loader.filedict['I']['img']) == 2

def test_data_loader_generate_filedict_with_num_samples():
    """Simple test to check generating file dict specifying a limited number 
    of files."""

    test_data_dirpath = 'tests/deepdisc/test_data/'

    #this block is for debug purposes, set to -1 to include every sample
    num_samples = 1

    # Initialize the DDLoader object
    hsc_loader = DDLoader()

    filters = ['I']

    # Generate the dictionary of file paths
    hsc_loader.generate_filedict(
        os.path.abspath(os.path.join(test_data_dirpath, 'hsc')),
        filters,
        '*_scarlet_img.fits',
        '*_scarlet_segmask.fits',
        n_samples=num_samples
    )

    hsc_loader.filedict['filters']
    assert hsc_loader.filedict['filters'] == filters
    assert len(hsc_loader.filedict['I']['img']) == 1

def test_data_loader_generate_filedict_with_subdir():
    """Simple test to check generating file dict using a directory structure that
    includes subdirectories."""

    test_data_dirpath = 'tests/deepdisc/test_data/'

    # Initialize the DDLoader object
    hsc_loader = DDLoader()

    filters = ['I']

    # Generate the dictionary of file paths
    hsc_loader.generate_filedict(
        os.path.abspath(os.path.join(test_data_dirpath, 'hsc')),
        filters,
        '*_scarlet_img.fits',
        '*_scarlet_segmask.fits',
        subdirs=True
    )

    hsc_loader.filedict['filters']
    assert hsc_loader.filedict['filters'] == filters
    assert len(hsc_loader.filedict['I']['img']) == 1

def test_data_loader_generate_dataset_dict_hsc():
    """Simple test for hcs dataset dict generation"""

    test_data_dirpath = 'tests/deepdisc/test_data/'

    # Initialize the DDLoader object
    hsc_loader = DDLoader()

    filters = ['G', 'R', 'I']

    num_samples = 1

    # Generate the dictionary of file paths
    hsc_loader.generate_filedict(
        os.path.abspath(os.path.join(test_data_dirpath, 'hsc')),
        filters,
        '*_scarlet_img.fits',
        '*_scarlet_segmask.fits',
        n_samples=num_samples,
    )

    hsc_loader.filedict['filters']
    assert hsc_loader.filedict['filters'] == filters
    assert len(hsc_loader.filedict['G']['img']) == 1
    assert len(hsc_loader.filedict['R']['img']) == 1
    assert len(hsc_loader.filedict['I']['img']) == 1

    hsc_loader.generate_dataset_dict(annotate_hsc)

    dataset_dict = hsc_loader.get_dataset()

    assert len(dataset_dict[0]['annotations']) == 552

def test_data_loader_generate_dataset_no_file_dict():
    """Test assertion when no file dict is present"""

    # Initialize the DDLoader object
    hsc_loader = DDLoader()

    with pytest.raises(ValueError) as excinfo:
        hsc_loader.generate_dataset_dict(annotate_hsc)
        assert "No file dictionary" in excinfo.value

def test_data_loader_generate_filedict_raises_with_unequal_file_numbers():
    """Test expects an exception to be raised if when there are unequal numbers
    of files per filter."""

    test_data_dirpath = 'tests/deepdisc/test_data/'

    # Initialize the DDLoader object
    hsc_loader = DDLoader()

    filters = ['G', 'R', 'I']

    # Generate the dictionary of file paths
    with pytest.raises(RuntimeError) as excinfo:
        hsc_loader.generate_filedict(
            os.path.abspath(os.path.join(test_data_dirpath, 'hsc')),
            filters,
            '*_scarlet_img.fits',
            '*_scarlet_segmask.fits'
        )
        assert "Found different number" in excinfo.value
