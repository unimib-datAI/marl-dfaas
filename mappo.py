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
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    if hasattr(policy, "compute_central_vf"):
        assert other_agent_batches is not None
        if policy.config["enable_connectors"]:
            [(_, _, opponent_batch)] = list(other_agent_batches.values())
        else:
            [(_, opponent_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(
                convert_to_torch_tensor(sample_batch[SampleBatch.CUR_OBS], policy.device),
                convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
                convert_to_torch_tensor(sample_batch[OPPONENT_ACTION], policy.device),
            ).cpu().detach().numpy()
        )
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(sample_batch,
                                     last_r,
                                     policy.config["gamma"],
                                     policy.config["lambda"],
                                     use_gae=policy.config["use_gae"])
    return train_batch


class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        return centralized_critic_postprocessing(self, sample_batch, other_agent_batches, episode)


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

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = obs_space.shape[0] * 2 + action_space.dim # obs + opp_obs + opp_act
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 16, activation_fn=nn.Tanh),
            SlimFC(16, 1),
        )
    
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model.forward(input_dict, state, seq_lens)
        return model_out, []
    
    def central_value_function(self, obs, opponent_obs, opponent_actions):
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
