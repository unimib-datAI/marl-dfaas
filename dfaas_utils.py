import sys
from pathlib import Path
import tomllib
import gzip

import orjson


def dict_to_json(data, file_path):
    """Serializes a dictionary to a JSON file using orjson."""
    file_path = to_pathlib(file_path)

    # Since orjson does not support serializing Path objects, manually
    # convert those values to plain strings.
    #
    # We assume that these values can only appear at the top level of the
    # dictionary.
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], Path):
                data[key] = data[key].as_posix()

    try:
        with open(file_path, "wb") as file:
            enc = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SORT_KEYS)
            file.write(enc)
            file.write(b"\n")
    except IOError as e:
        print(
            f"Failed to write dict to json file to {file_path.as_posix()!r}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def toml_to_dict(file_path):
    """Loads a TOML file into a dictionary."""
    file_path = to_pathlib(file_path)

    try:
        with open(file_path, "rb") as file:
            return tomllib.load(file)
    except IOError as e:
        print(
            f"Failed to read toml file from {file_path.as_posix()!r}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def json_to_dict(file_path):
    """Loads a JSON file into a dictionary."""
    file_path = to_pathlib(file_path)

    try:
        with open(file_path, "r") as file:
            return orjson.loads(file.read())
    except IOError as e:
        print(
            f"Failed to read json file from {file_path.as_posix()!r}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def to_pathlib(file_path):
    """Returns an (absolute) Path object from the given path."""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    return file_path.absolute()


def parse_result_file(result_path):
    """Parses a result file containing JSONL objects.

    If the file ends with '.gz', it is opened using gzip.

    Args:
        result_path (str or Path): The path to the result file.

    Returns:
        list: A list where each item is a JSON-decoded object from a line.
    """
    result_path = to_pathlib(result_path)

    if result_path.name.endswith(".gz"):
        open_func = gzip.open
        mode = "rt"
    else:
        open_func = open
        mode = "r"

    # The "result.json(.gz)" file is not a valid JSON file. Each line is an
    # isolated JSON object, the result of one training iteration.
    with open_func(result_path, mode) as result:
        return [orjson.loads(line) for line in result]
