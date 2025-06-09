"""This module executes a training experiment using a supported algorithm with
the DFaaS environment."""

from pathlib import Path
import shutil
from datetime import datetime
import logging
import argparse

import tqdm

import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.registry import get_policy_class
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models.catalog import MODEL_DEFAULTS

import dfaas_utils
import dfaas_env
import dfaas_apl

# Disable Ray's warnings.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logger for this module.
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(Path(__file__).name)


def main():
    epilog = """Some command line options can override the configuration of
    --exp-config option, if provided."""
    description = "Run a training experiment on the DFaaS environment."

    parser = argparse.ArgumentParser(prog="dfaas_train", description=description, epilog=epilog)

    parser.add_argument(dest="suffix", help="A string to append to experiment directory name")
    parser.add_argument("--exp-config", type=Path, help="Override default experiment config (TOML file)")
    parser.add_argument("--env-config", type=Path, help="Override default environment config (TOML file)")
    parser.add_argument("--runners", type=int, help="Number of parallel runners to play episodes")
    parser.add_argument("--seed", type=int, help="Seed of the experiment")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after one training iteration (useful for debugging purposes). Default is False",
    )

    args = parser.parse_args()

    # Initialize the experiment configuration from exp-config argument.
    if args.exp_config is not None:
        exp_config = dfaas_utils.toml_to_dict(args.exp_config)
    else:
        exp_config = {}

    # Override experiment configuration from CLI arguments. Only a limited
    # subset of options can be overrode.
    override_options = ["env_config", "runners", "seed"]
    for option in override_options:
        if getattr(args, option) is not None:
            # The argparse module always sets the argument to None if not
            # provided.
            exp_config[option] = getattr(args, option)

    # Set default experiment configuration values if not provided.
    exp_config["iterations"] = exp_config.get("iterations", 100)
    exp_config["disable_gpu"] = exp_config.get("disable_gpu", False)
    exp_config["runners"] = exp_config.get("runners", 1)
    exp_config["seed"] = exp_config.get("seed", 42)
    exp_config["algorithm"] = exp_config.get("algorithm", "PPO")
    exp_config["checkpoint_interval"] = exp_config.get("checkpoint_interval", 50)
    exp_config["evaluation_interval"] = exp_config.get("evaluation_interval", 50)
    exp_config["final_evaluation"] = exp_config.get("final_evaluation", True)
    exp_config["env"] = dfaas_env.DFaaS.__name__

    logger.info(f"Experiment configuration")
    for key, value in exp_config.items():
        logger.info(f"{key:>25}: {value}")

    # Environment configuration.
    if exp_config.get("env_config") is not None:
        env_config = dfaas_utils.toml_to_dict(exp_config["env_config"])
    else:
        env_config = {}

    # Create a dummy environment, used as reference.
    dummy_env = dfaas_env.DFaaS(config=env_config)
    logger.info(f"Environment configuration")
    for key, value in dummy_env.get_config().items():
        logger.info(f"{key:>25}: {value}")

    # For the evaluation phase at the end, the env_config is different than the
    # training one.
    env_eval_config = env_config.copy()
    env_eval_config["evaluation"] = True

    ray.init(include_dashboard=False)

    # PolicySpec is required to specify the action/observation space for each
    # policy. In this case, each policy has the same spaces.
    #
    # See this thread: https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/5
    #
    # Each agent has its own policy (policy_node_X in policy -> node_X in the env).
    #
    # Note that if no option is given to PolicySpec, it will inherit the
    # configuration/algorithm from the main configuration.
    policies = {}
    policies_to_train = []
    for agent in dummy_env.agents:
        policy_name = f"policy_{agent}"
        policies[policy_name] = PolicySpec(
            policy_class=None,
            observation_space=dummy_env.observation_space[agent],
            action_space=dummy_env.action_space[agent],
            config=None,
        )
        policies_to_train.append(policy_name)

    # Allow to overwrite the default policies with the TOML config file.
    if exp_config.get("policies") is not None:
        policies.clear()
        policies_to_train.clear()

        for policy_cfg in exp_config["policies"]:
            policy_name = policy_cfg["name"]

            # PolicySpec expects the Python class of the policy, not the raw
            # string!
            if policy_cfg.get("class"):
                assert policy_cfg["class"] == "APLPolicy", "Only APLPolicy is supported as custom policy"
                policy_class = dfaas_apl.APLPolicy
            else:
                # It uses PPO (default if None).
                policy_class = None
                policies_to_train.append(policy_name)

            policies[policy_name] = PolicySpec(
                policy_class=policy_class,
                observation_space=dummy_env.observation_space[agent],
                action_space=dummy_env.action_space[agent],
                config=None,
            )

        assert len(policies) == len(dummy_env.agents), "Each policy should be mapped to an agent (and viceversa)"

    # Log policies data.
    for policy_name, policy in policies.items():
        logger.info(f"Policy: name = {policy_name!r}, class = {policy.policy_class!r}")
    logger.info(f"Policies to train: {policies_to_train}")

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        """Called by RLlib at each step to map an agent to a policy (defined above).
        In this case, the map is static: every agent has the same policy, and a
        policy has the same single agent."""
        return f"policy_{agent_id}"

    # Model options.
    model = MODEL_DEFAULTS.copy()
    # Must be False to have a different network for the Critic.
    model["vf_share_layers"] = False

    if exp_config.get("model") is not None:
        # Update the model with the given options.
        given_model = dfaas_utils.json_to_dict(exp_config.get("model"))
        model = model | given_model

    assert dummy_env.max_steps == 288, "Only 288 steps supported for the environment"

    match exp_config["algorithm"]:
        case "PPO":
            experiment = build_ppo(
                runners=exp_config["runners"],
                dummy_env=dummy_env,
                env_config=env_config,
                model=model,
                env_eval_config=env_eval_config,
                exp_config=exp_config,
                no_gpu=False,
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=policies_to_train,
            )
        case "SAC":
            # WARNING: SAC support is experimental in the DFaaS environment. It
            # doesn't work very well because I have to tune the hyperparameters,
            # the neural network and the sampling method.
            experiment = build_sac(
                runners=exp_config["runners"],
                dummy_env=dummy_env,
                env_config=env_config,
                model=model,
                env_eval_config=env_eval_config,
                exp_config=exp_config,
                no_gpu=False,
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
        case "APL":
            experiment = build_apl(
                runners=exp_config["runners"],
                dummy_env=dummy_env,
                env_config=env_config,
                env_eval_config=env_eval_config,
                exp_config=exp_config,
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
        case _:
            raise ValueError(f"Algorithm {exp_config['algorithm']!r} not found")

    # Get the experiment directory to save other files.
    logdir = Path(experiment.logdir).resolve()
    logger.info(f"DFAAS experiment directory created at {logdir.as_posix()!r}")
    # This will be used after the evaluation.
    start = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_file = logdir / "exp_config.json"
    dfaas_utils.dict_to_json(exp_config, exp_file)
    logger.info(f"Experiment configuration saved to: {exp_file.as_posix()!r}")

    dummy_config = dummy_env.get_config()
    env_config_path = logdir / "env_config.json"
    dfaas_utils.dict_to_json(dummy_config, env_config_path)
    logger.info(f"Environment configuration saved to: {env_config_path.as_posix()!r}")

    # Save the model architecture for all policies (if the algorithm provides
    # one).
    if exp_config["algorithm"] != "APL":
        policies_dir = logdir / "policy_model"
        policies_dir.mkdir()
        for policy_name, policy in experiment.env_runner.policy_map.items():
            out = policies_dir / policy_name
            out.write_text(f"{policy.model}")
            logger.info(f"Policy '{policy_name}' model saved to: {out.as_posix()!r}")

    # Copy the environment source file (and its associated code) into the
    # experiment directory. This ensures that the original environment used for
    # the experiment is preserved.
    for filename in ["dfaas_env.py", "perfmodel.py", "dfaas_input_rate.py"]:
        src_path = Path.cwd() / filename
        dst_path = logdir / filename

        try:
            shutil.copy2(src_path, dst_path)
            logger.info(f"Environment source file {filename!r} copied to {dst_path.as_posix()!r}")
        except FileNotFoundError:
            logger.warning(f"Failed to copy {filename!r}: file not found")

    # Each item is an evaluation result dictionary. This will be saved to the
    # disk at the end of the experiment.
    eval_result = []
    eval_file = logdir / "evaluation.json"

    # Create the experiment name. At the end of the experiment this name will be
    # used as output directory name.
    exp_name = f"DF_{start}_{exp_config['algorithm']}_{args.suffix}"
    logger.info(f"Experiment name: {exp_name}")

    # Run the training phase.
    max_iterations = exp_config["iterations"]
    assert max_iterations > 0, "Iterations must be a positive number!"
    checkpoint_interval = exp_config["checkpoint_interval"]
    assert checkpoint_interval >= 0, "Checkpoint interval must be non negative!"
    evaluation_interval = exp_config["evaluation_interval"]
    assert evaluation_interval >= 0, "Evaluation interval must be non negative!"
    logger.info("Training start")
    dry_run = args.dry_run
    with tqdm.tqdm(total=max_iterations) as progress_bar:
        for iteration in range(max_iterations):
            experiment.train()

            # Exclude checkpoint and evaluation from progress bar timings.
            progress_bar.update(0)

            if dry_run:
                break

            # Save a checkpoint every checkpoint_interval iterations (0 means
            # disabled).
            if checkpoint_interval > 0 and ((iteration + 1) % checkpoint_interval) == 0:
                checkpoint_name = f"checkpoint_{iteration:04d}"
                checkpoint_path = (logdir / checkpoint_name).as_posix()
                experiment.save(checkpoint_path)
                logger.info(f"Checkpoint {checkpoint_name!r} saved")

            # Evaluate every evaluate_interval iterations (0 means no
            # evaluation).
            if evaluation_interval > 0 and ((iteration + 1) % evaluation_interval) == 0:
                logger.info(f"Evaluation of the {iteration}-th iteration")
                evaluation = experiment.evaluate()
                evaluation["iteration"] = iteration
                eval_result.append(evaluation)

            progress_bar.update(1)

    # Save always the latest training iteration.
    latest_iteration = exp_config["iterations"] - 1
    checkpoint_path = logdir / f"checkpoint_{latest_iteration:03d}"
    if not checkpoint_path.exists():  # May exist if max_iter is a multiple of 50.
        checkpoint_path = checkpoint_path.as_posix()
        experiment.save(checkpoint_path)
        logger.info(f"Final checkpoint saved to {checkpoint_path!r}")
    logger.info(f"Training results data saved to: {experiment.logdir}/result.json")

    # Do a final evaluation.
    if exp_config["final_evaluation"]:
        logger.info("Evaluation of the final iteration")
        evaluation = experiment.evaluate()
        evaluation["iteration"] = latest_iteration
        eval_result.append(evaluation)

    # Save the evaluation data (if present).
    if len(eval_result) > 0:
        dfaas_utils.dict_to_json(eval_result, eval_file)
        logger.info(f"Evaluation results data saved to: {eval_file.as_posix()}")
    else:
        logger.info(f"Evaluation results empty, skip saving")

    # Remove unused or problematic files in the result directory.
    Path(logdir / "progress.csv").unlink()  # result.json contains same data.
    Path(logdir / "params.json").unlink()  # params.pkl contains same data (and the JSON is broken).

    # Move the original experiment directory to a custom directory.
    result_dir = Path.cwd() / "results" / exp_name
    shutil.move(logdir, result_dir.resolve())
    logger.info(f"DFAAS experiment results moved to {result_dir.as_posix()!r}")


def build_ppo(**kwargs):
    runners = kwargs["runners"]
    dummy_env = kwargs["dummy_env"]
    env_config = kwargs["env_config"]
    model = kwargs["model"]
    env_eval_config = kwargs["env_eval_config"]
    exp_config = kwargs["exp_config"]
    no_gpu = kwargs["no_gpu"]
    policies = kwargs["policies"]
    policy_mapping_fn = kwargs["policy_mapping_fn"]
    policies_to_train = kwargs["policies_to_train"]

    # The train_batch_size is the total number of samples to collect for each
    # iteration across all runners. Since the user can specify 0 runners, we must
    # ensure that we collect at least 288 samples (1 complete episodes).
    #
    # Be careful with train_batch_size: RLlib stops the episodes when this number is
    # reached, it doesn't control each runner. The number should be divisible by the
    # number of runners, otherwise a runner has to collect more (or less) samples
    # and plays one plus or minus episode.
    #
    # In my case I just let each runner to play one episode.
    episodes_per_iter = 1 * (runners if runners > 0 else 1)
    train_batch_size = dummy_env.max_steps * episodes_per_iter

    # Keep the default number of SGD iterations but reduce the minibatch size,
    # since we play only one episode. This may let the agent to overfit, but for
    # now it is ok.
    #
    # Note that even if we have a minibatch size of 64, since the environment
    # length is not a multiple, the last batch will have a smaller size.
    num_epochs = 30
    minibatch_size = 64

    config = (
        PPOConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        # For each iteration, store only the episodes calculated in that
        # iteration in the log result.
        .reporting(metrics_num_episodes_for_smoothing=episodes_per_iter)
        .environment(env=dfaas_env.DFaaS.__name__, env_config=env_config)
        .training(train_batch_size=train_batch_size, num_epochs=num_epochs, minibatch_size=minibatch_size, model=model)
        .framework("torch")
        # Wait max 4 minutes for each iteration to collect the samples.
        .env_runners(num_env_runners=runners, sample_timeout_s=240)
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=10,
            evaluation_num_env_runners=1,
            evaluation_config={"env_config": env_eval_config},
        )
        .debugging(seed=exp_config["seed"])
        .resources(num_gpus=0 if no_gpu else 1)
        .callbacks(dfaas_env.DFaaSCallbacks)
        .multi_agent(policies=policies, policies_to_train=policies_to_train, policy_mapping_fn=policy_mapping_fn)
    )

    return config.build()


def build_sac(**kwargs):
    runners = kwargs["runners"]
    dummy_env = kwargs["dummy_env"]
    env_config = kwargs["env_config"]
    model = kwargs["model"]
    env_eval_config = kwargs["env_eval_config"]
    exp_config = kwargs["exp_config"]
    no_gpu = kwargs["no_gpu"]
    policies = kwargs["policies"]
    policy_mapping_fn = kwargs["policy_mapping_fn"]

    # Replay buffer configuration.
    # Other values are set to the default.
    replay_buffer_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 4 * int(1e5),
        "num_shards": 1,  # Each agent has it is own buffer of full capacity.
        # "num_shards": len(dummy_env.agents)  # TODO
    }

    # In each iteration, each runner plays exactly three complete episodes.
    episodes_iter = 3 * (runners if runners > 0 else 1)
    episodes_runner = 3
    rollout_fragment_length = dummy_env.max_steps * episodes_runner

    # Collect random exploratory experience before starting training, to help
    # initialize the replay buffer.
    warm_up_size = int(1e4)

    # In each iteration, we do 12 mini-batch update epochs on the models. Each
    # batch has a size of 256.
    #
    # Formula:
    #
    #   native_ratio = train_batch_size / (rollout_fragment_length * max(runners + 1, 1))
    #
    #   epochs = training_intensity / native_ratio
    #
    # See calculate_rr_weights() in rllib/algorithms/dqn/dqn.py
    train_batch_size = 256
    training_intensity = 0.6

    config = (
        SACConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        # For each iteration, store only the episodes calculated in that
        # iteration in the log result.
        .reporting(metrics_num_episodes_for_smoothing=episodes_iter)
        .environment(env=dfaas_env.DFaaS.__name__, env_config=env_config)
        .training(
            num_steps_sampled_before_learning_starts=warm_up_size,
            training_intensity=training_intensity,
            train_batch_size=train_batch_size,
            policy_model_config=model,
            replay_buffer_config=replay_buffer_config,
        )
        .framework("torch")
        .env_runners(rollout_fragment_length=rollout_fragment_length, num_env_runners=runners)
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=10,
            evaluation_num_env_runners=1,
            evaluation_config={"env_config": env_eval_config},
        )
        .debugging(seed=exp_config["seed"])
        .resources(num_gpus=0 if no_gpu else 1)
        .callbacks(dfaas_env.DFaaSCallbacks)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    )

    return config.build()


def build_apl(**kwargs):
    runners = kwargs["runners"]
    dummy_env = kwargs["dummy_env"]
    env_config = kwargs["env_config"]
    env_eval_config = kwargs["env_eval_config"]
    exp_config = kwargs["exp_config"]
    policies = kwargs["policies"]
    policy_mapping_fn = kwargs["policy_mapping_fn"]

    # See build_ppo function.
    episodes_iter = 1 * (runners if runners > 0 else 1)
    train_batch_size = dummy_env.max_steps * episodes_iter

    config = (
        dfaas_apl.APLConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        # For each iteration, store only the episodes calculated in that
        # iteration in the log result.
        .reporting(metrics_num_episodes_for_smoothing=episodes_iter)
        .environment(env=dfaas_env.DFaaS.__name__, env_config=env_config)
        .training(train_batch_size=train_batch_size)
        .env_runners(num_env_runners=runners)
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=10,
            evaluation_num_env_runners=1,
            evaluation_config={"env_config": env_eval_config},
        )
        .debugging(seed=exp_config["seed"])
        .callbacks(dfaas_env.DFaaSCallbacks)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    )

    return config.build()


if __name__ == "__main__":
    main()
