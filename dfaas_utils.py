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
    file_path = to_pathlib(file_path)

    try:
        with open(file_path, "w") as file:
            json.dump(data, file, cls=NumpyEncoder)
    except IOError as e:
        print(f"Failed to write dict to json file to {file_path.as_posix()!r}: {e}",
              file=sys.stderr)
        sys.exit(1)


def json_to_dict(file_path):
    file_path = to_pathlib(file_path)

    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except IOError as e:
        print(f"Failed to read json file from {file_path.as_posix()!r}: {e}",
              file=sys.stderr)
        sys.exit(1)


def to_pathlib(file_path):
    # Make sure to have a Path object, because we want the absolute path.
    if isinstance(file_path, str):
        file_path = Path(file_path)
    return file_path.absolute()


def parse_result_file(result_path):
    result_path = to_pathlib(result_path)

    # Fill the iters list with the "result.json" file.
    iters = []
    with result_path.open() as result:
        # The "result.json" file is not a valid JSON file. Each row is an
        # isolated JSON object, the result of one training iteration.
        while (raw_iter := result.readline()) != "":
            iters.append(json.loads(raw_iter))

    return iters


