from os import path
import pytest

TEST_DIR = path.dirname(__file__)

@pytest.fixture
def hsc_test_data_dir():
    return path.join(TEST_DIR, "test_data/hsc")

@pytest.fixture
def hsc_single_test_file(hsc_test_data_dir):
    return path.join(hsc_test_data_dir, "single_test.json")
