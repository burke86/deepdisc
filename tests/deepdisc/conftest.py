from os import path
import pytest

TEST_DIR = path.dirname(__file__)

@pytest.fixture
def hsc_test_data_dir():
    return path.join(TEST_DIR, "test_data/hsc")

@pytest.fixture
def hsc_single_test_file(hsc_test_data_dir):
    return path.join(hsc_test_data_dir, "single_test.json")

@pytest.fixture
def hsc_triple_test_file(hsc_test_data_dir):
    return [
        path.join(hsc_test_data_dir, "G-10054-0,2-c1_scarlet_img.fits"),
        path.join(hsc_test_data_dir, "I-10054-0,2-c1_scarlet_img.fits"),
        path.join(hsc_test_data_dir, "R-10054-0,2-c1_scarlet_img.fits"),
    ]

@pytest.fixture
def dc2_test_data_dir():
    return path.join(TEST_DIR, "test_data/dc2")

@pytest.fixture
def dc2_single_test_file(dc2_test_data_dir):
    return path.join(dc2_test_data_dir, "3828_2,2_12_images.npy")


@pytest.fixture
def dc2_single_test_dict(dc2_test_data_dir):
    return path.join(dc2_test_data_dir, "single_test.json")