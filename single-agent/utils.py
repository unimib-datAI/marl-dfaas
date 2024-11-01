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


class OutputDuplication():
    """A sys.stdout duplicator: it writes data to both sys.stdout and a file.

    The class is partly compatible with io.TextIOBase. After instantiation, all
    calls to "write" will output messages to sys.stdout and to a buffer. When
    "set_logfile()" is called, the buffer is flushed to the file and subsequent
    calls to "write()" output the messages to sys.stdout and to the file.
    """

    def __init__(self):
        self.terminal = sys.stdout
        self.logfile = None
        self.buffer = []
        self.encoding = "utf-8"  # Required by io.TextIOBase

    def write(self, message):
        """See io.TextIOBase for more information."""
        self.terminal.write(message)

        if self.logfile is None:
            self.buffer.append(message)
        else:
            self.logfile.write(message)

    def set_logfile(self, logfile):
        file = open(logfile, mode="w", encoding="utf-8")

        file.writelines(self.buffer)
        self.logfile = file
        self.buffer.clear()

    def flush(self):
        """See io.IOBase for more information."""
        self.terminal.flush()
        if self.logfile is not None:
            self.logfile.flush()

    def fileno(self):
        """See io.IOBase for more information."""
        return self.terminal.fileno()

    def __del__(self):
        """Closes the underlying file."""
        if self.logfile is not None:
            self.logfile.close()
        self.logfile = None
