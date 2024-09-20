from pathlib import Path
import logging

import dfaas_env
import dfaas_utils

# Reset the matplotlib logger to warning, because Ray (called by dfaas_env
# module) changes the level to debug.
_matplotlib_logger = logging.root.manager.loggerDict["matplotlib"]
_matplotlib_logger.setLevel("WARNING")

# Initialize logger for this module.
logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s", level=logging.DEBUG)
logger = logging.getLogger(Path(__file__).name)

# This module-level variable holds the environment class.
_env = None


def _env_init(exp_dir):
    """Initializes the internal _env variable by creating the environment with
    the class (DFaaS, DFaaS_ASYM...) and configuration extracted from the given
    experiment directory."""
    if not exp_dir.exists():
        logger.critical(f"Experiment directory not found: {exp_dir.as_posix()!r}")
        raise FileNotFoundError(exp_dir)

    # Experiment configuration (read the existing one).
    exp_config = dfaas_utils.json_to_dict(exp_dir / "exp_config.json")

    # Try to load the environment class from the dfaas_env module.
    try:
        DFaaS = getattr(dfaas_env, exp_config["env"])
    except AttributeError:
        logger.critical(f"Environment {exp_config['env']!r} not found in dfaas_env.py")
        raise Exception

    # Environment configuration (read the existing one).
    env_config = dfaas_utils.json_to_dict(exp_dir / "env_config.json")

    # Create the environment with the given env config.
    global _env
    _env = DFaaS(config=env_config)


def get_env(exp_dir):
    """Returns the environment from the given experiment directory."""
    if _env is None:
        _env_init(exp_dir)

    return _env
