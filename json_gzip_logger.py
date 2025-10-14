from pathlib import Path
import gzip
import json
from typing import override

from ray.tune.logger import Logger
from ray.tune.utils.util import SafeFallbackEncoder


class JsonGzipLogger(Logger):
    """
    A logger that writes results to a gzip-compressed JSON file.

    I use this logger instead of the standard one because the JSON data can be
    quite large, and compressing it helps reduce the file size.

    By default, in Ray RLlib's old stack API, a UnifiedLogger is used with the
    default configuration to log results in multiple formats, including plain
    JSON, CSV, TensorBoardX files, and a few additional minor files. The
    UnifiedLogger is essentially a combination of several specialized loggers,
    such as CsvLogger and JsonLogger.

    In addition, the Algorithm class extends the Trainable class, and the latter
    is responsible for creating the default logger when no specific options are
    provided. This is why the logger interface, along with classes like
    UnifiedLogger and JsonLogger, are located under the ray.train module.

    Args:
        config (dict): Configuration dictionary for the logger.
        logdir (str or Path): Directory where the log file will be saved.

    Attributes:
        file (gzip.GzipFile): The gzip-compressed file object used for logging.

    References:
        https://github.com/ray-project/ray/blob/master/python/ray/tune/logger/json.py#L30
        https://github.com/ray-project/ray/blob/master/python/ray/tune/logger/logger.py#L35
        https://github.com/ray-project/ray/blob/master/python/ray/tune/logger/unified.py#L19
        https://github.com/ray-project/ray/blob/master/python/ray/tune/trainable/trainable.py#L666
        https://github.com/ray-project/ray/blob/master/rllib/algorithms/algorithm.py#L207
    """

    def __init__(self, config, logdir):
        super().__init__(config, logdir)

        log_path = Path(logdir) / "result.json.gz"
        self.file = gzip.open(log_path, "wt", encoding="utf-8")

    @override
    def on_result(self, result):
        # The SafeFallbackEncoder class comes from Ray and has additional
        # handling for objects that cannot be encoded with the default encoder.
        json.dump(result, self.file, cls=SafeFallbackEncoder)
        self.file.write("\n")

        # We noted that Ray RLlib does not call close() when an experiment
        # terminates, so we need to continuously flush data to disk with every
        # write.
        self.flush()

    @override
    def update_config(self, config):
        # This method is unused but required by the parent class.
        pass

    @override
    def close(self):
        self.file.close()

    @override
    def flush(self):
        self.file.flush()
