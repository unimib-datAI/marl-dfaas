# This Python script is expected to be imported by all Jupyter notebooks.
from pathlib import Path
import logging
import sys
import os

import IPython.core.debugger

import matplotlib

# Add the parent directory (the root directory of the project) to sys.path. This
# is needed to load modules like dfaas_utils or dfaas_env.
sys.path.append(Path(os.getcwd()).parent.as_posix())

import dfaas_env
import dfaas_utils

# Reset the matplotlib logger to warning, because Ray (called by dfaas_env
# module) changes the level to debug.
_matplotlib_logger = logging.root.manager.loggerDict["matplotlib"]
_matplotlib_logger.setLevel("WARNING")

# Initialize logger for this module.
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(Path(__file__).name)

# This module-level variable holds the environment object. Supports multiple
# experiments (the key is the experiment path, value is the object).
_env = dict()


def _env_init(exp_dir):
    """Initializes the internal _env variable by creating the environment with
    the configuration extracted from the given experiment directory."""
    if not exp_dir.exists():
        logger.critical(f"Experiment directory not found: {exp_dir.as_posix()!r}")
        raise FileNotFoundError(exp_dir)

    # Experiment configuration (read the existing one).
    exp_config = dfaas_utils.json_to_dict(exp_dir / "exp_config.json")

    # Environment configuration (read the existing one).
    env_config = dfaas_utils.json_to_dict(exp_dir / "env_config.json")

    # Create the environment with the given env config.
    global _env
    _env[exp_dir] = dfaas_env.DFaaS(config=env_config)


def get_env(exp_dir):
    """Returns the environment from the given experiment directory."""
    if exp_dir not in _env:
        _env_init(exp_dir)

    return _env[exp_dir]


def set_trace():
    """Open IPython debugger in interactive-mode."""
    debugger.set_trace()


# Default size (10) is too small.
font = {"size": 12}
matplotlib.rcParams["figure.figsize"] = [9, 6]
matplotlib.rc("font", **font)
