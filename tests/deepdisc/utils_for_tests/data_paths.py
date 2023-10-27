from os import path


def get_test_hsc_data_path(file_or_directory):
    """Construct the absolute file path of the test HSC data.

    Parameters
    ----------
    file_or_directory: str
        The name of the file or directory within the HSC data to access.

    Returns
    -------
    full_path: str
        The absolute path name for the resource.
    """
    # The test directory is the one level outside where this file resides.
    test_dir = path.abspath(path.dirname(path.dirname(__file__)))

    # Add in the subdirectory.
    data_dir = path.join(test_dir, "test_data/hsc")

    # Add in the file or directory.
    full_path = path.join(data_dir, file_or_directory)
    return full_path


def get_test_dc2_data_path(file_or_directory):
    """Construct the absolute file path of the test DC2 data.

    Parameters
    ----------
    file_or_directory: str
        The name of the file or directory within the HSC data to access.

    Returns
    -------
    full_path: str
        The absolute path name for the resource.
    """
    # The test directory is the one level outside where this file resides.
    test_dir = path.abspath(path.dirname(path.dirname(__file__)))

    # Add in the subdirectory.
    data_dir = path.join(test_dir, "test_data/dc2")

    # Add in the file or directory.
    full_path = path.join(data_dir, file_or_directory)
    return full_path
