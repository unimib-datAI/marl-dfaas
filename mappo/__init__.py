from .centralized_critic_model import CustomTorchCCModel, CCPPOTorchPolicy
from .algorithm import MAPPOConfig

from ray.rllib.algorithms.registry import (
  ALGORITHMS_CLASS_TO_NAME,
  POLICIES,
  ALGORITHMS
)
from ray.rllib.models import ModelCatalog

def _import_mappo():
  import mappo as mappo
  return mappo.MAPPO, mappo.MAPPO.get_default_config()

ALGORITHMS_CLASS_TO_NAME["MAPPO"] = "MAPPO"
ALGORITHMS["MAPPO"] = _import_mappo
POLICIES["CCPPOTorchPolicy"] = CCPPOTorchPolicy

ModelCatalog.register_custom_model("centralizedcritic", CustomTorchCCModel)
