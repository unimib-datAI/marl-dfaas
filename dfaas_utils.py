import json
import sys
from pathlib import Path

import numpy as np


# Thanks to: https://stackoverflow.com/a/47626762
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()

        # Use the default JSON encoder for other types.
        return json.JSONEncoder.default(self, obj)


def dict_to_json(data, file_path):
    # Make sure to have a Path object, because we want the absolute path.
    if isinstance(file_path, str):
        file_path = Path(file_path)
    file_path = file_path.absolute()

    try:
        with open(file_path, "w") as file:
            json.dump(data, file, cls=NumpyEncoder)
    except IOError as e:
        print(f"Failed to write dict to json file to {file_path.as_posix()!r}: {e}",
              file=sys.stderr)
        sys.exit(1)


def json_to_dict(file_path):
    # Make sure to have a Path object, because we want the absolute path.
    if isinstance(file_path, str):
        file_path = Path(file_path)
    file_path = file_path.absolute()

    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except IOError as e:
        print(f"Failed to read json file from {file_path.as_posix()!r}: {e}",
              file=sys.stderr)
        sys.exit(1)
