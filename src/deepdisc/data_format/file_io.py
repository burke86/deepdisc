import json


def get_data_from_json(file):
    """Open a JSON text file, and return encoded data as dictionary.

    Parameters
    ----------
    file : str
        pointer to file

    Returns
    -------
        dictionary of encoded data
    """
    # Opening JSON file
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
