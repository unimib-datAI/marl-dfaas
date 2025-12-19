# from ray.rllib.examples._old_api_stack.models.centralized_critic_models import YetAnotherTorchCentralizedCriticModel as TorchCentralizedCriticModel
from ray.rllib.examples._old_api_stack.models.centralized_critic_models import TorchCentralizedCriticModel
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import SlimFC

torch, nn = try_import_torch()
import numpy as np


OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"


def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
  # save original value function
  vf_saved = model.value_function
  # calculate loss with a custom value function
  model.value_function = lambda: policy.model.central_value_function(
    train_batch[SampleBatch.CUR_OBS],
    train_batch[OPPONENT_OBS],
    train_batch[OPPONENT_ACTION],
  )
  policy._central_value_out = model.value_function()
  loss = base_policy.loss(model, dist_class, train_batch)
  # restore original value function
  model.value_function = vf_saved
  return loss


class CentralizedValueMixin:
  """Add method to evaluate the central value function from the model."""
  def __init__(self):
    self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
  ):
  if hasattr(policy, "compute_central_vf"):
    assert other_agent_batches is not None
    if policy.config["enable_connectors"]:
      # values() -> list of (other_policy, _, batch)
      opponent_batches = [
        triple[-1] for triple in other_agent_batches.values()
      ]
    else:
      # values() -> list of (other_policy, batch)
      opponent_batches = [pair[-1] for pair in other_agent_batches.values()]
    # separate lists of observations and actions
    op_obs_list = [b[SampleBatch.CUR_OBS] for b in opponent_batches]
    op_act_list = [b[SampleBatch.ACTIONS] for b in opponent_batches]
    # # average
    # opponent_obs = np.mean(np.stack(op_obs_list, axis=0), axis=0)
    # opponent_actions = np.mean(np.stack(op_act_list, axis=0), axis=0)
    # concatenate along feature dimension
    opponent_obs = np.concatenate(op_obs_list, axis=1)     # [B, N * obs_dim]
    opponent_actions = np.concatenate(op_act_list, axis=1) # [B, N * act_dim]
    # record the opponent obs and actions in the trajectory
    sample_batch[OPPONENT_OBS] = opponent_obs
    sample_batch[OPPONENT_ACTION] = opponent_actions
    # overwrite default VF prediction with the central VF
    sample_batch[SampleBatch.VF_PREDS] = (
      policy.compute_central_vf(
        convert_to_torch_tensor(
          sample_batch[SampleBatch.CUR_OBS], policy.device
        ),
        convert_to_torch_tensor(
          sample_batch[OPPONENT_OBS], policy.device
        ),
        convert_to_torch_tensor(
          sample_batch[OPPONENT_ACTION], policy.device
        ),
      ).cpu().detach().numpy()
    )
  else:
    n_agents = policy.config["model"]["custom_model_config"]["n_agents"]
    # policy hasn't been initialized yet, use zeros.
    sample_batch[OPPONENT_OBS] = np.zeros_like(
      np.concatenate(
        [
          sample_batch[SampleBatch.CUR_OBS] for _ in range(n_agents - 1)
        ], 
        axis = 1
      )
    )
    sample_batch[OPPONENT_ACTION] = np.zeros_like(
      np.concatenate(
        [
          sample_batch[SampleBatch.ACTIONS] for _ in range(n_agents - 1)
        ],
        axis = 1
      )
    )
    sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
      sample_batch[SampleBatch.REWARDS], dtype=np.float32
    )
  # record if completed
  completed = sample_batch[SampleBatch.TERMINATEDS][-1]
  if completed:
    last_r = 0.0
  else:
    last_r = sample_batch[SampleBatch.VF_PREDS][-1]
  # compute advantage
  train_batch = compute_advantages(
    sample_batch,
    last_r,
    policy.config["gamma"],
    policy.config["lambda"],
    use_gae=policy.config["use_gae"]
  )
  return train_batch


class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
  def __init__(self, observation_space, action_space, config):
    PPOTorchPolicy.__init__(self, observation_space, action_space, config)
    CentralizedValueMixin.__init__(self)

  def loss(self, model, dist_class, train_batch):
    return loss_with_central_critic(
      self, super(), model, dist_class, train_batch
    )

  def postprocess_trajectory(
      self, sample_batch, other_agent_batches=None, episode=None
    ):
    return centralized_critic_postprocessing(
      self, sample_batch, other_agent_batches, episode
    )


class MAPPO(PPO):
  @classmethod
  def get_default_policy_class(cls, config):
    return CCPPOTorchPolicy


class MAPPOConfig(PPOConfig):
  def __init__(self, algo_class=None):
    super().__init__(algo_class=algo_class or MAPPO)


class CustomTorchCCModel(TorchCentralizedCriticModel):
  def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    super().__init__(
      obs_space, action_space, num_outputs, model_config, name
    )
    # central VF maps (obs, opp_obs, opp_act) -> vf_pred
    n_agents = model_config.get("custom_model_config", {}).get("n_agents", 2)
    input_size = obs_space.shape[0] * n_agents + action_space.dim * (n_agents - 1) # obs + opp_obs + opp_act
    self.central_vf = nn.Sequential(
      SlimFC(input_size, 16, activation_fn=nn.Tanh),
      SlimFC(16, 1),
    )
  
  def forward(self, input_dict, state, seq_lens):
    model_out, _ = self.model.forward(input_dict, state, seq_lens)
    return model_out, []
  
  def central_value_function(self, obs, opponent_obs, opponent_actions):
    # obs:             [B, obs_dim]
    # opponent_obs:    [B, N*obs_dim] or [B, obs_dim] (if averaged)
    # opponent_actions:[B, N*act_dim] or [B, act_dim]
    input_ = torch.cat(
      [
        obs,
        opponent_obs,
        opponent_actions,
      ],
      1,
    )
    return torch.reshape(self.central_vf(input_), [-1])


ModelCatalog.register_custom_model("centralizedcritic", CustomTorchCCModel)
