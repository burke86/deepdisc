import json
from pathlib import Path


def get_data_from_json(filename):
    """Open a JSON text file, and return encoded data as dictionary.

    Parameters
    ----------
    filename : str
        The name of the file to load.

    Returns
    -------
        dictionary of encoded data

    Raises
    ------
    FileNotFoundError if the file cannot be found.
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"Unable to load file {filename}")

    # Opening JSON file
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
