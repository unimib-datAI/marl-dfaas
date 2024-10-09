from pathlib import Path
import shutil
from datetime import datetime
import logging
import argparse

import numpy as np

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.examples.models.centralized_critic_models import TorchCentralizedCriticModel

import dfaas_utils
import dfaas_env

# Disable Ray's warnings.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(prog="dfaas_run_ppo")
parser.add_argument(dest="env", help="DFaaS environment to train")
parser.add_argument(dest="suffix", help="A string to append to experiment directory")
parser.add_argument("--no-gpu",
                    help="Disable GPU usage",
                    default=True, dest="gpu", action="store_false")
parser.add_argument("--env-config", help="Environment config file")
args = parser.parse_args()

# Initialize logger for this module.
logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s", level=logging.DEBUG)
logger = logging.getLogger(Path(__file__).name)

# Try to load the given environment class from the dfaas_env module.
try:
    DFaaS = getattr(dfaas_env, args.env)
    logger.info(f"Environment {DFaaS.__name__!r} loaded")
except AttributeError:
    logger.error(f"Environment {args.env!r} not found in dfaas_env.py")
    exit(1)

# The number of rollout workers is fixed to five. This is because 4320 steps are
# collected in each iteration, 864 from each worker (3 episodes).
rollout_workers = 5

# Experiment configuration.
# TODO: make this configurable!
exp_config = {"seed": 42,  # Seed of the experiment.
              "max_iterations": 500,  # Number of iterations.
              "env": DFaaS.__name__,  # Environment.
              "gpu": args.gpu,
              "workers": rollout_workers
              }
logger.info(f"Experiment configuration = {exp_config}")

# Env configuration.
if args.env_config is not None:
    env_config = dfaas_utils.json_to_dict(args.env_config)
else:
    env_config = {}
logger.info(f"Environment configuration = {env_config}")

# For the evaluation phase at the end, the env_config is different than the
# training one.
env_eval_config = env_config.copy()
env_eval_config["evaluation"] = True

# Create a dummy environment, used to get observation and action spaces.
dummy_env = DFaaS(config=env_config)

# PolicySpec is required to specify the action/observation space for each
# policy. Because each agent in the env has different action and observation
# space, it is important to configure them.
#
# See this thread: https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/5
#
# Each agent has its own policy (policy_node_X in policy -> node_X in the env).
#
# Note that if no option is given to PolicySpec, it will inherit the
# configuration/algorithm from the main configuration.
policies = {"policy_node_0": PolicySpec(policy_class=None,
                                        observation_space=dummy_env.observation_space["node_0"],
                                        action_space=dummy_env.action_space["node_0"],
                                        config=None),
            "policy_node_1": PolicySpec(policy_class=None,
                                        observation_space=dummy_env.observation_space["node_1"],
                                        action_space=dummy_env.action_space["node_1"],
                                        config=None)
            }


# This function is called by Ray to determine which policy to use for an agent
# returned in the observation dictionary by step() or reset().
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''This function is called at each step to assign the agent to a policy. In
    this case, each agent has a fixed corresponding policy.'''
    return f"policy_{agent_id}"


OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"


# Copied from PPO but optimizing the central value function.
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


class CentralizedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
        return CCPPOTorchPolicy


ModelCatalog.register_custom_model("centralized_model", TorchCentralizedCriticModel)

assert dummy_env.max_steps == 288, "Only 288 steps supported for the environment"

# Each rollout worker plays 3 episodes. Since max_steps and rollout_workers are
# fixed, the result is 4320.
train_batch_size = 3 * dummy_env.max_steps * rollout_workers

# Algorithm config.
ppo_config = (PPOConfig()
              .environment(env=DFaaS.__name__, env_config=env_config)
              .training(train_batch_size=train_batch_size,
                        model={"custom_model": "centralized_model"})
              .framework("torch")
              .rollouts(num_rollout_workers=rollout_workers)
              .evaluation(evaluation_interval=None, evaluation_duration=50,
                          evaluation_num_workers=1,
                          evaluation_config={"env_config": env_eval_config})
              .debugging(seed=exp_config["seed"])
              .resources(num_gpus=1 if args.gpu else 0)
              .callbacks(dfaas_env.DFaaSCallbacks)
              .multi_agent(policies=policies,
                           policy_mapping_fn=policy_mapping_fn)
              )

# Build the experiment.
ppo_algo = ppo_config.build()

# Get the experiment directory to save other files.
logdir = Path(ppo_algo.logdir).resolve()
logger.info(f"DFAAS experiment directory created at {logdir.as_posix()!r}")
# This will be used after the evaluation.
start = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

exp_file = logdir / "exp_config.json"
dfaas_utils.dict_to_json(exp_config, exp_file)
logger.info(f"Experiment configuration saved to: {exp_file.as_posix()!r}")

dummy_config = dummy_env.get_config()
env_config_path = logdir / "env_config.json"
dfaas_utils.dict_to_json(dummy_config, env_config_path)
logger.info(f"Environment configuration saved to: {env_config_path.as_posix()!r}")

# Copy the environment source file into the experiment directory. This ensures
# that the original environment used for the experiment is preserved.
dfaas_env_dst = logdir / Path(dfaas_env.__file__).name
shutil.copy2(dfaas_env.__file__, dfaas_env_dst)
logger.info(f"Environment source file saved to: {dfaas_env_dst.as_posix()!r}")

# Run the training phase for n iterations.
logger.info("Training start")
for iteration in range(exp_config["max_iterations"]):
    logger.info(f"Iteration {iteration}")
    ppo_algo.train()

    # Save a checkpoint every 50 iterations.
    if ((iteration + 1) % 50) == 0:
        checkpoint_path = (logdir / f"checkpoint_{iteration:03d}").as_posix()
        ppo_algo.save(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path!r}")

# Save always the latest training iteration.
latest_iteration = exp_config["max_iterations"] - 1
checkpoint_path = logdir / f"checkpoint_{latest_iteration:03d}"
if not checkpoint_path.exists():  # May exist if max_iter is a multiple of 50.
    checkpoint_path = checkpoint_path.as_posix()
    ppo_algo.save(checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path!r}")
logger.info(f"Iterations data saved to: {ppo_algo.logdir}/result.json")

# Do a final evaluation.
logger.info("Final evaluation start")
evaluation = ppo_algo.evaluate()
eval_file = logdir / "evaluation.json"
dfaas_utils.dict_to_json(evaluation, eval_file)
logger.info(f"Final evaluation saved to: {ppo_algo.logdir}/final_evaluation.json")

# Remove this file as it is redundant, "result.json" already contains the same
# data.
Path(logdir / "progress.csv").unlink()

# Move the original experiment directory to a custom directory.
exp_name = f"DFAAS-MA_{start}_{DFaaS.type}_{args.suffix}"
result_dir = Path.cwd() / "results" / exp_name
shutil.move(logdir, result_dir.resolve())
logger.info(f"DFAAS experiment results moved to {result_dir.as_posix()!r}")
