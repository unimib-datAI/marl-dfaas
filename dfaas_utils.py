import sys
from pathlib import Path

import orjson


def dict_to_json(data, file_path):
    file_path = to_pathlib(file_path)

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


def json_to_dict(file_path):
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
    # Make sure to have a Path object, because we want the absolute path.
    if isinstance(file_path, str):
        file_path = Path(file_path)
    return file_path.absolute()


def parse_result_file(result_path):
    result_path = to_pathlib(result_path)

    # The "result.json" file is not a valid JSON file. Each line is an isolated
    # JSON object, the result of one training iteration.
    with result_path.open() as result:
        return [orjson.loads(line) for line in result]
