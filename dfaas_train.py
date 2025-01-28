# Main Python script to run PPO or SAC algorithm training experiment with DFaaS
# environment.
from pathlib import Path
import shutil
from datetime import datetime
import logging
import argparse

from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models.catalog import MODEL_DEFAULTS

import dfaas_utils
import dfaas_env

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
    parser = argparse.ArgumentParser(
        prog="dfaas_train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="suffix", help="A string to append to experiment directory"
    )
    parser.add_argument(
        "--no-gpu",
        help="Disable GPU usage",
        default=False,
        dest="no_gpu",
        action="store_true",
    )
    parser.add_argument("--env-config", help="Environment config file")
    parser.add_argument(
        "--iterations",
        default=500,
        type=int,
        help="Number of iterations to run (non-negative integer)",
    )
    parser.add_argument(
        "--algorithm", default="PPO", choices=["PPO", "SAC"], help="Algorithm to use"
    )
    parser.add_argument(
        "--runners",
        default=5,
        type=int,
        help="Number of runners for collecting experencies in each iteration",
    )
    parser.add_argument(
        "--model", type=Path, help="Override default neural networks model"
    )
    parser.add_argument(
        "--skip-evaluation",
        default=False,
        action="store_true",
        help="Skip final evaluation",
    )

    args = parser.parse_args()

    # By default, there are five runners collecting samples, each running 3 complete
    # episodes (for a total of 4320 samples, 864 for each runner).
    #
    # The number of runners can be changed. Each runner is a process. If set to
    # zero, sampling is done on the main process.
    runners = args.runners

    # Experiment configuration.
    # TODO: make this configurable!
    # TODO: Add custom model option.
    exp_config = {
        "seed": 42,  # Seed of the experiment.
        "max_iterations": args.iterations,  # Number of iterations.
        "env": dfaas_env.DFaaS.__name__,  # Environment.
        "gpu": not args.no_gpu,
        "final_evaluation": not args.skip_evaluation,
        "runners": runners,
    }
    logger.info(f"Experiment configuration = {exp_config}")

    # Environment configuration.
    if args.env_config is not None:
        env_config = dfaas_utils.json_to_dict(args.env_config)
    else:
        env_config = {}

    # Create a dummy environment, used as reference.
    dummy_env = dfaas_env.DFaaS(config=env_config)
    logger.info(f"Environment configuration = {dummy_env.get_config()}")

    # For the evaluation phase at the end, the env_config is different than the
    # training one.
    env_eval_config = env_config.copy()
    env_eval_config["evaluation"] = True

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
    for agent in dummy_env.agents:
        policy_name = f"policy_{agent}"
        policies[policy_name] = PolicySpec(
            policy_class=None,
            observation_space=dummy_env.observation_space[agent],
            action_space=dummy_env.action_space[agent],
            config=None,
        )

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        """Called by RLlib at each step to map an agent to a policy (defined above).
        In this case, the map is static: every agent has the same policy, and a
        policy has the same single agent."""
        return f"policy_{agent_id}"

    # Model options.
    model = MODEL_DEFAULTS.copy()
    # Must be False to have a different network for the Critic.
    model["vf_share_layers"] = False

    if args.model is not None:
        # Update the model with the given options.
        given_model = dfaas_utils.json_to_dict(args.model)
        model = model | given_model

    assert dummy_env.max_steps == 288, "Only 288 steps supported for the environment"

    match args.algorithm:
        case "PPO":
            experiment = build_ppo(
                runners=runners,
                dummy_env=dummy_env,
                env_config=env_config,
                model=model,
                env_eval_config=env_eval_config,
                exp_config=exp_config,
                no_gpu=args.no_gpu,
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
        case "SAC":
            # WARNING: SAC support is experimental in the DFaaS environment. It
            # doesn't work very well because I have to tune the hyperparameters,
            # the neural network and the sampling method.
            experiment = build_sac(
                runners=runners,
                dummy_env=dummy_env,
                env_config=env_config,
                model=model,
                env_eval_config=env_eval_config,
                exp_config=exp_config,
                no_gpu=args.no_gpu,
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
        case _:
            assert False, "Unreachable code"

    # Get the experiment directory to save other files.
    logdir = Path(experiment.logdir).resolve()
    logger.info(f"DFAAS experiment directory created at {logdir.as_posix()!r}")
    # This will be used after the evaluation.
    start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    exp_file = logdir / "exp_config.json"
    dfaas_utils.dict_to_json(exp_config, exp_file)
    logger.info(f"Experiment configuration saved to: {exp_file.as_posix()!r}")

    dummy_config = dummy_env.get_config()
    env_config_path = logdir / "env_config.json"
    dfaas_utils.dict_to_json(dummy_config, env_config_path)
    logger.info(f"Environment configuration saved to: {env_config_path.as_posix()!r}")

    # Save the model architecture for all policies.
    policies_dir = logdir / "policy_model"
    policies_dir.mkdir()
    for policy_name, policy in experiment.env_runner.policy_map.items():
        out = policies_dir / policy_name
        out.write_text(f"{policy.model}")
        logger.info(f"Policy '{policy_name}' model saved to: {out.as_posix()!r}")

    # Copy the environment source file into the experiment directory. This ensures
    # that the original environment used for the experiment is preserved.
    dfaas_env_dst = logdir / Path(dfaas_env.__file__).name
    shutil.copy2(dfaas_env.__file__, dfaas_env_dst)
    logger.info(f"Environment source file saved to: {dfaas_env_dst.as_posix()!r}")

    # Run the training phase for n iterations.
    logger.info("Training start")
    for iteration in range(exp_config["max_iterations"]):
        logger.info(f"Iteration {iteration}")
        result = experiment.train()

        # Save a checkpoint every 50 iterations.
        if ((iteration + 1) % 50) == 0:
            checkpoint_path = (logdir / f"checkpoint_{iteration:03d}").as_posix()
            experiment.save(checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path!r}")

    # Save always the latest training iteration.
    latest_iteration = exp_config["max_iterations"] - 1
    checkpoint_path = logdir / f"checkpoint_{latest_iteration:03d}"
    if not checkpoint_path.exists():  # May exist if max_iter is a multiple of 50.
        checkpoint_path = checkpoint_path.as_posix()
        experiment.save(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path!r}")
    logger.info(f"Iterations data saved to: {experiment.logdir}/result.json")

    # Do a final evaluation.
    if not args.skip_evaluation:
        logger.info("Final evaluation start")
        evaluation = experiment.evaluate()
        eval_file = logdir / "evaluation.json"
        dfaas_utils.dict_to_json(evaluation, eval_file)
        logger.info(
            f"Final evaluation saved to: {experiment.logdir}/final_evaluation.json"
        )

    # Remove unused or problematic files in the result directory.
    Path(logdir / "progress.csv").unlink()  # result.json contains same data.
    Path(
        logdir / "params.json"
    ).unlink()  # params.pkl contains same data (and the JSON is broken).

    # Move the original experiment directory to a custom directory.
    exp_name = f"DFAAS-MA_{start}_{args.algorithm}_{args.suffix}"
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

    # The train_batch_size is the total number of samples to collect for each
    # iteration across all runners. Since the user can specify 0 runners, we must
    # ensure that we collect at least 864 samples (3 complete episodes).
    #
    # Be careful with train_batch_size: RLlib stops the episodes when this number is
    # reached, it doesn't control each runner. The number should be divisible by the
    # number of runners, otherwise a runner has to collect more (or less) samples
    # and plays one plus or minus episode.
    episodes_iter = 3 * runners if runners > 0 else 1
    train_batch_size = dummy_env.max_steps * episodes_iter

    config = (
        PPOConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        # For each iteration, store only the episodes calculated in that
        # iteration in the log result.
        .reporting(metrics_num_episodes_for_smoothing=episodes_iter)
        .environment(env=dfaas_env.DFaaS.__name__, env_config=env_config)
        .training(train_batch_size=train_batch_size, model=model)
        .framework("torch")
        .env_runners(num_env_runners=runners)
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=50,
            evaluation_num_env_runners=1,
            evaluation_config={"env_config": env_eval_config},
        )
        .debugging(seed=exp_config["seed"])
        .resources(num_gpus=0 if no_gpu else 1)
        .callbacks(dfaas_env.DFaaSCallbacks)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
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
        "capacity": 10**6,
    }

    # episodes_iter = 3 * runners if runners > 0 else 1

    # Play one episode for each iteration.
    rollout_fragment_length = dummy_env.max_steps

    config = (
        SACConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        # For each iteration, store only the episodes calculated in that
        # iteration in the log result.
        # .reporting(metrics_num_episodes_for_smoothing=episodes_iter)
        .environment(env=dfaas_env.DFaaS.__name__, env_config=env_config)
        .training(
            policy_model_config=model,
            replay_buffer_config=replay_buffer_config,
        )
        .framework("torch")
        .env_runners(
            rollout_fragment_length=rollout_fragment_length, num_env_runners=runners
        )
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=50,
            evaluation_num_env_runners=1,
            evaluation_config={"env_config": env_eval_config},
        )
        .debugging(seed=exp_config["seed"])
        .resources(num_gpus=0 if no_gpu else 1)
        .callbacks(dfaas_env.DFaaSCallbacks)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    )

    return config.build()


if __name__ == "__main__":
    main()