"""This module executes a training experiment using a supported algorithm with
the DFaaS environment."""

from pathlib import Path
import shutil
from datetime import datetime
import logging
import argparse

import tqdm
import yaml

import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models.catalog import MODEL_DEFAULTS

from json_gzip_logger import JsonGzipLogger
import dfaas_utils
import dfaas_apl
import dfaas_env
from dfaas_env_config import DFaaSConfig

# Disable Ray's warnings.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logger for this module.
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(Path(__file__).name)


def run_experiment(
    suffix,
    exp_config=None,
    env_config=None,
    runners=None,
    seed=None,
    dry_run=False,
):
    # The experiment and environment configs are now passed as dicts from main().
    # Override experiment configuration from input arguments. Only a limited
    # subset of options can be overrode.
    if exp_config is None:
        exp_config = {}

    override_options = ["env_config", "runners", "seed"]
    override_values = {
        "env_config": env_config,
        "runners": runners,
        "seed": seed,
    }
    for option in override_options:
        if override_values[option] is not None:
            exp_config[option] = override_values[option]

    # Set default experiment configuration values if not provided. See
    # configs/exp/ppo.yaml for docs.
    exp_config["iterations"] = exp_config.get("iterations", 100)
    exp_config["disable_gpu"] = exp_config.get("disable_gpu", False)
    exp_config["training_num_episodes"] = exp_config.get("training_num_episodes", 1)
    exp_config["runners"] = exp_config.get("runners", 1)
    exp_config["seed"] = exp_config.get("seed", 42)
    exp_config["algorithm"] = exp_config.get("algorithm", dict())
    exp_config["algorithm"]["name"] = exp_config["algorithm"].get("name", "PPO")
    exp_config["algorithm"]["gamma"] = exp_config["algorithm"].get("gamma", 0.99)
    if exp_config["algorithm"]["name"] == "PPO":
        exp_config["algorithm"]["lambda"] = exp_config["algorithm"].get("lambda", 1)
        exp_config["algorithm"]["entropy_coeff"] = exp_config["algorithm"].get("entropy_coeff", 0)
        exp_config["algorithm"]["entropy_coeff_decay_enable"] = exp_config["algorithm"].get(
            "entropy_coeff_decay_enable", False
        )
        exp_config["algorithm"]["entropy_coeff_decay_iterations"] = exp_config["algorithm"].get(
            "entropy_coeff_decay_iterations", 0.7
        )
    exp_config["checkpoint_interval"] = exp_config.get("checkpoint_interval", 50)
    exp_config["evaluation_interval"] = exp_config.get("evaluation_interval", 50)
    exp_config["evaluation_num_episodes"] = exp_config.get("evaluation_num_episodes", 10)
    exp_config["final_evaluation"] = exp_config.get("final_evaluation", True)

    logger.info("Experiment configuration")
    for key, value in exp_config.items():
        logger.info(f"{key:>25}: {value}")

    # Environment configuration.
    # FIXME: Save the env config inside exp config before be dumped to JSON.
    if exp_config.get("env_config") is not None:
        # The env_config is now passed as a dict from main().
        env_config = exp_config["env_config"]
    else:
        env_config = {}

    # Create a dummy environment, used as reference. This will also validate the
    # given configuration.
    dummy_env = DFaaSConfig.from_dict(env_config).build()
    logger.info("Environment configuration loaded and validated!")

    # For the evaluation phase at the end, the env_config is different than the
    # training one.
    env_eval_config = env_config.copy()
    env_eval_config["evaluation"] = True
    env_eval_config["seed"] = exp_config["seed"]

    # Ray RLlib requires a Ray cluster to run. We first check whether we're
    # already connected to one. If not, we need to connect before starting the
    # experiment.
    if not ray.is_initialized():
        try:
            # Try to connect to an existing Ray cluster.
            ray.init(address="auto", include_dashboard=False)
        except Exception:
            logger.warning("Failed to connect to Ray cluster, creating a local one")
            # Start a local Ray instance.
            ray.init(include_dashboard=False)
    ray_address = ray._private.worker._global_node.address_info["address"]
    logger.info(f"Ray address: {ray_address}")

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

    # Allow to overwrite the default policies with the YAML config file.
    if exp_config.get("policies") is not None:
        policies.clear()
        policies_to_train.clear()

        for policy_cfg in exp_config["policies"]:
            policy_name = policy_cfg["name"]

            # PolicySpec expects the Python class of the policy, not the raw
            # string!
            if policy_cfg.get("class"):
                if policy_cfg["class"] != "APLPolicy":
                    raise ValueError("Only APLPolicy is supported as custom policy")
                policy_class = dfaas_apl.APLPolicy
            else:
                # It uses the default algorithm.
                policy_class = None
                policies_to_train.append(policy_name)

            policies[policy_name] = PolicySpec(
                policy_class=policy_class,
                observation_space=dummy_env.observation_space[agent],
                action_space=dummy_env.action_space[agent],
                config=None,
            )

        if len(policies) != len(dummy_env.agents):
            raise ValueError("Each policy should be mapped to an agent (and viceversa)")

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
    # Default is just two hidden layers with 256 neurons each.
    model["fcnet_hiddens"] = [256, 256, 256, 256]

    if exp_config.get("model") is not None:
        # Update the model with the given options.
        given_model = dfaas_utils.json_to_dict(exp_config.get("model"))
        model = model | given_model

    if dummy_env.max_steps != 288:
        raise ValueError("Only 288 steps supported for the environment")

    match exp_config["algorithm"]["name"]:
        case "PPO":
            experiment = build_ppo(
                runners=exp_config["runners"],
                dummy_env=dummy_env,
                env_config=env_config,
                model=model,
                env_eval_config=env_eval_config,
                seed=exp_config["seed"],
                no_gpu=False,
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=policies_to_train,
                evaluation_num_episodes=exp_config["evaluation_num_episodes"],
                training_num_episodes=exp_config["training_num_episodes"],
                gamma=exp_config["algorithm"]["gamma"],
                lambda_=exp_config["algorithm"]["lambda"],
                entropy_coeff=exp_config["algorithm"]["entropy_coeff"],
                entropy_coeff_decay_enable=exp_config["algorithm"]["entropy_coeff_decay_enable"],
                entropy_coeff_decay_iterations=exp_config["algorithm"]["entropy_coeff_decay_iterations"],
                max_iterations=exp_config["iterations"],
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
                seed=exp_config["seed"],
                no_gpu=False,
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=policies_to_train,
                evaluation_num_episodes=exp_config["evaluation_num_episodes"],
                training_num_episodes=exp_config["training_num_episodes"],
                gamma=exp_config["algorithm"]["gamma"],
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
            raise ValueError(f"Algorithm {exp_config['algorithm']['name']!r} not found")

    # Get the experiment directory to save other files.
    logdir = Path(experiment.logdir).resolve()
    logger.info(f"DFAAS experiment directory created at {logdir.as_posix()!r}")
    # This will be used after the evaluation.
    start = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_start = datetime.now()

    exp_file = logdir / "exp_config.json"
    dfaas_utils.dict_to_json(exp_config, exp_file)
    logger.info(f"Experiment configuration saved to: {exp_file.as_posix()!r}")

    # Save environment configuration to disk as YAML file.
    env_config_path = logdir / "env_config.yaml"
    env_config_path.write_text(yaml.dump(dummy_env.get_config().to_dict(), sort_keys=True, indent=4))
    logger.info(f"Environment configuration saved to: {env_config_path.as_posix()!r}")

    # Save the model architecture for all policies (if the algorithm provides
    # one).
    if exp_config["algorithm"]["name"] != "APL":
        policies_dir = logdir / "policy_model"
        policies_dir.mkdir()
        for policy_name, policy in experiment.env_runner.policy_map.items():
            out = policies_dir / policy_name
            out.write_text(f"{policy.model}")
            logger.info(f"Policy '{policy_name}' model saved to: {out.as_posix()!r}")

    # Copy the environment source file (and its associated code) into the
    # experiment directory. This ensures that the original environment used for
    # the experiment is preserved.
    for filename in [
        "dfaas_env.py",
        "dfaas_env_config.py",
        "bandwidth_generator.py",
        "perfmodel.py",
        "dfaas_input_rate.py",
    ]:
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
    exp_name = f"DF_{start}_{exp_config['algorithm']['name']}_{suffix}"
    logger.info(f"Experiment name: {exp_name}")

    # Prepare the evaluation environment for each evaluation runner.
    #
    # The following code calls self.set_master_seed once for each evaluation
    # runner. However, we only have one evaluation runner. We also know the
    # number of evaluation episodes for each iteration.
    #
    # This is necessary to ensure that each evaluation iteration uses the same
    # generation seeds. Ray is not flexible in this regard, which is why I need
    # to write this hack: pre-generate all the seeds for one iteration and cycle
    # through them within each iteration. See the DFaaS env for more
    # information.
    evaluation_num_episodes = exp_config["evaluation_num_episodes"]
    if evaluation_num_episodes <= 0:
        raise ValueError("At least one eval. episodes must be run")

    def reset_env(env):
        master_seed = env.master_seed
        env.set_master_seed(master_seed, evaluation_num_episodes)

    def reset_env_worker(worker):
        worker.foreach_env(reset_env)

    experiment.eval_env_runner_group.foreach_worker(reset_env_worker)

    # Run the training phase.
    max_iterations = exp_config["iterations"]
    if max_iterations <= 0:
        raise ValueError("Iterations must be a positive number!")
    checkpoint_interval = exp_config["checkpoint_interval"]
    if checkpoint_interval < 0:
        raise ValueError("Checkpoint interval must be non negative!")
    evaluation_interval = exp_config["evaluation_interval"]
    if evaluation_interval < 0:
        raise ValueError("Evaluation interval must be non negative!")
    logger.info(f"Training start ({max_iterations} iterations)")
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
                checkpoint_name = f"checkpoint_{iteration:05d}"
                checkpoint_path = logdir / checkpoint_name
                experiment.save(checkpoint_path.as_posix())
                logger.info(f"Checkpoint {checkpoint_name!r} saved")

            # Evaluate every evaluate_interval iterations (0 means no
            # evaluation).
            if evaluation_interval > 0 and ((iteration + 1) % evaluation_interval) == 0:
                logger.info(f"Evaluation of the {iteration}-th iteration")
                evaluation = experiment.evaluate()
                evaluation["iteration"] = iteration
                eval_result.append(evaluation)

            progress_bar.update(1)

    # We can get the latest done iteration from the for ... range loop. Note tha
    # can be different from exp_config["iterations"] - 1, since we can make
    # fewer iterations (like with the --dry-run flag).
    last_iteration = iteration

    # Save always a checkpoint for the last training iteration.
    checkpoint_name = f"checkpoint_{last_iteration:05d}"
    checkpoint_path = logdir / checkpoint_name
    if checkpoint_path.exists():
        # The last checkpoint may already exists due to checkpoint_interval.
        logger.info(f"Final checkpoint is {checkpoint_name!r} (already saved)")
    else:
        experiment.save(checkpoint_path.as_posix())
        logger.info(f"Checkpoint {checkpoint_name!r} saved")
    logger.info(f"Training results data saved to: {experiment.logdir}/result.json.gz")

    # Do a final evaluation.
    if exp_config["final_evaluation"]:
        if len(eval_result) > 0 and eval_result[-1]["iteration"] == last_iteration:
            # Evaluation already done due to evaluation_interval.
            logger.info(f"Final evaluation is already done ({last_iteration}-th iter.)")
        else:
            logger.info(f"Evaluation of the final ({last_iteration}-th) iteration")
            evaluation = experiment.evaluate()
            evaluation["iteration"] = last_iteration
            eval_result.append(evaluation)

    # Save the evaluation data (if present).
    if len(eval_result) > 0:
        dfaas_utils.dict_to_json(eval_result, eval_file)
        logger.info(f"Evaluation results data saved to: {eval_file.as_posix()}")
    else:
        logger.info("Evaluation results empty, skip saving")

    # Move the original experiment directory to a custom directory.
    result_dir = Path.cwd() / "results" / exp_name
    shutil.move(logdir, result_dir.resolve())
    logger.info(f"DFAAS experiment results moved to {result_dir.as_posix()!r}")

    experiment_end = datetime.now()
    experiment_duration = experiment_end - experiment_start
    logger.info(f"Experiment name: {exp_name}")
    logger.info(f"Experiment duration: {experiment_duration}")

    return {
        "experiment_name": exp_name,
        "result_dir": result_dir,
        "experiment_duration": experiment_duration,
        "eval_result": eval_result,
    }


def build_ppo(**kwargs):
    runners = kwargs["runners"]
    dummy_env = kwargs["dummy_env"]
    env_config = kwargs["env_config"]
    model = kwargs["model"]
    env_eval_config = kwargs["env_eval_config"]
    seed = kwargs["seed"]
    no_gpu = kwargs["no_gpu"]
    policies = kwargs["policies"]
    policy_mapping_fn = kwargs["policy_mapping_fn"]
    policies_to_train = kwargs["policies_to_train"]
    evaluation_num_episodes = kwargs["evaluation_num_episodes"]
    training_num_episodes = kwargs["training_num_episodes"]
    gamma = kwargs["gamma"]
    lambda_ = kwargs["lambda_"]
    entropy_coeff = kwargs["entropy_coeff"]
    entropy_coeff_decay_enable = kwargs["entropy_coeff_decay_enable"]
    entropy_coeff_decay_iterations = kwargs["entropy_coeff_decay_iterations"]
    max_iterations = kwargs["max_iterations"]

    if not 0 <= gamma <= 1:
        raise ValueError("Gamma (discount factor) must be between 0 and 1")
    if not 0 <= lambda_ <= 1:
        raise ValueError("Lambda must be between 0 and 1")

    # Checks for the training_num_episodes and runners parameters.
    if training_num_episodes <= 0:
        raise ValueError("Must play at least one episode for each iteration!")
    if runners < 0:
        raise ValueError("The number of runners must be non-negative!")
    if runners == 0:
        episodes_per_runner = training_num_episodes
    else:
        episodes_per_runner, remainder = divmod(training_num_episodes, runners)
        if remainder != 0:
            raise ValueError(f"Each runner must play the same number of episodes ({remainder} != 0)!")

    # In PPO, the entropy coefficient allows us to balance exploration and
    # exploitation by encouraging the former. In our case, we have a continuous
    # action space modeled with a Dirichlet distribution, which already has a
    # natural built-in spread. However, the entropy coefficient can still
    # provide additional help.
    #
    # By default, if no value is provided, the coefficient remains constant at
    # 0. If specified, we apply a slow decay over the first X% of iterations. We
    # only define the start and end values, intermediate timesteps are assigned
    # linearly interpolated coefficient values. Note that the timestep in the
    # schedule is global (the sum of all agents' timesteps) rather than specific
    # to each individual agent.
    #
    # References:
    #   - https://github.com/ray-project/ray/blob/master/rllib/policy/torch_mixins.py#L51
    #   - https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo.py#L277
    #   - https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo_torch_policy.py#L153
    #   - https://github.com/ray-project/ray/blob/master/rllib/evaluation/rollout_worker.py#L1620
    if entropy_coeff != 0.0 and entropy_coeff_decay_enable:
        # Decay for a custom percentage of iterations (counting timesteps).
        if not (0 < entropy_coeff_decay_iterations <= 1):
            raise ValueError("entropy_coeff_decay_iterations must be in (0, 1]")

        end_timestep = entropy_coeff_decay_iterations * (
            dummy_env.max_steps * len(dummy_env.agents) * episodes_per_runner * max_iterations
        )
        entropy_coeff_schedule = [[0, entropy_coeff], [end_timestep, 1e-6]]
    else:
        entropy_coeff_schedule = None

    # The train_batch_size is the total number of samples to collect for each
    # iteration across all runners. Since the user can specify 0 runners, we
    # must ensure that we collect at least 288 samples (1 complete episode).
    #
    # Be careful with train_batch_size: RLlib stops the episodes when this
    # number is reached, it doesn't control each runner. The number must be
    # divisible by the number of runners, otherwise a runner has to collect more
    # (or less) samples and plays one plus or minus episode.
    train_batch_size = dummy_env.max_steps * training_num_episodes

    # Run 30 SGD iteration over the training batch (num_epochs). For each
    # iteration, the training batch is divided in minibatches with defined size
    # (minibatch_size), so the number of updates can vary. The minibatch size
    # should divide the training batch, to have all equal-size batches.
    #
    # Note: if the minibatch size does not divide the training batch size, the
    # last minibatch will have a smaller size.
    num_epochs = 30
    minibatch_size = 144

    updates_per_epoch = train_batch_size // minibatch_size
    total_gradient_updates = num_epochs * updates_per_epoch

    config = (
        PPOConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        # For each iteration, store only the episodes calculated in that
        # iteration in the log result.
        .reporting(metrics_num_episodes_for_smoothing=training_num_episodes)
        .environment(env=dfaas_env.DFaaS.__name__, env_config=env_config)
        .training(
            gamma=gamma,
            train_batch_size=train_batch_size,
            num_epochs=num_epochs,
            minibatch_size=minibatch_size,
            model=model,
            lambda_=lambda_,
            entropy_coeff=entropy_coeff,
            entropy_coeff_schedule=entropy_coeff_schedule,
        )
        .framework("torch")
        # Wait max 4 minutes for each iteration to collect the samples.
        .env_runners(num_env_runners=runners, sample_timeout_s=240)
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=evaluation_num_episodes,
            evaluation_num_env_runners=1 if evaluation_num_episodes > 0 else 0,
            evaluation_config={"env_config": env_eval_config},
        )
        .debugging(seed=seed, logger_config={"type": JsonGzipLogger})
        .resources(num_gpus=0 if no_gpu else 1)
        .callbacks(dfaas_env.DFaaSCallbacks)
        .multi_agent(policies=policies, policies_to_train=policies_to_train, policy_mapping_fn=policy_mapping_fn)
    )

    # Add custom properties to the algoritm configuration, these will be dumped
    # in the result.json file, useful for debugging after an experiment has
    # done.
    config.custom_properties = {}
    config.custom_properties["episodes_per_runner"] = episodes_per_runner
    config.custom_properties["updates_per_epoch"] = updates_per_epoch
    config.custom_properties["total_gradient_updates"] = total_gradient_updates

    return config.build()


def build_sac(**kwargs):
    runners = kwargs["runners"]
    dummy_env = kwargs["dummy_env"]
    env_config = kwargs["env_config"]
    model = kwargs["model"]
    env_eval_config = kwargs["env_eval_config"]
    seed = kwargs["seed"]
    no_gpu = kwargs["no_gpu"]
    policies = kwargs["policies"]
    policy_mapping_fn = kwargs["policy_mapping_fn"]
    policies_to_train = kwargs["policies_to_train"]
    evaluation_num_episodes = kwargs["evaluation_num_episodes"]
    training_num_episodes = kwargs["training_num_episodes"]
    gamma = kwargs["gamma"]

    if not 0 <= gamma <= 1:
        raise ValueError("Gamma (discount factor) must be between 0 and 1")

    # Assume 288 steps because I want to set buffer'size accordingly.
    if dummy_env.max_steps != 288:
        raise ValueError("SAC supports only environments with 288 steps")

    if training_num_episodes <= 0:
        raise ValueError("Must play at least one episode for each iteration!")

    # Replay buffer configuration (other values are set to the default).
    # By default save the latest 10 episodes' data in the buffer.
    replay_buffer_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": dummy_env.max_steps * 10,
        "num_shards": 1,  # Each agent has it is own buffer of full capacity.
    }

    # Fill the replay buffer before to start the agents' training.
    warm_up_size = replay_buffer_config["capacity"]

    if runners < 0:
        raise ValueError("The number of runners must be non-negative!")
    if runners == 0:
        episodes_per_runner = training_num_episodes
        runners = 1  # Same as 0, just one single runner.
    else:
        episodes_per_runner, remainder = divmod(training_num_episodes, runners)
        if remainder != 0:
            raise ValueError(f"Each runner must play the same number of episodes ({remainder} != 0)!")

    # In every training iteration collects all expected steps before training
    # starts. This ensures to wait all runners to finish.
    min_sample_timesteps_per_iteration = training_num_episodes * dummy_env.max_steps

    # Ensure only complete episodes are collected.
    batch_mode = "complete_episodes"

    # Each runner collects the same number of episodes, given as steps.
    rollout_fragment_length = dummy_env.max_steps * episodes_per_runner

    # For SAC, the train batch size is built by sampling the replay buffer. So
    # we should always have a value equal or greater than the collected data for
    # one iteration.
    train_batch_size = rollout_fragment_length * 3

    # WIP
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
    # train_batch_size = 256
    # training_intensity = 0.6

    config = (
        SACConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        # For each iteration, store only the episodes calculated in that
        # iteration in the log result.
        .reporting(metrics_num_episodes_for_smoothing=training_num_episodes)
        .environment(env=dfaas_env.DFaaS.__name__, env_config=env_config)
        .training(
            num_steps_sampled_before_learning_starts=warm_up_size,
            train_batch_size=train_batch_size,
            policy_model_config=model,
            replay_buffer_config=replay_buffer_config,
            gamma=gamma,
        )
        .reporting(min_sample_timesteps_per_iteration=min_sample_timesteps_per_iteration)
        .framework("torch")
        .env_runners(
            rollout_fragment_length=rollout_fragment_length,
            num_env_runners=runners,
            batch_mode=batch_mode,
            sample_timeout_s=240,  # Wait max 4 minutes for each runners.
        )
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=evaluation_num_episodes,
            evaluation_num_env_runners=1 if evaluation_num_episodes > 0 else 0,
            evaluation_config={"env_config": env_eval_config},
        )
        .debugging(seed=seed, logger_config={"type": JsonGzipLogger})
        .resources(num_gpus=0 if no_gpu else 1)
        .callbacks(dfaas_env.DFaaSCallbacks)
        .multi_agent(policies=policies, policies_to_train=policies_to_train, policy_mapping_fn=policy_mapping_fn)
    )

    # See build_ppo().
    config.custom_properties = {}
    config.custom_properties["episodes_per_runner"] = episodes_per_runner

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
        .debugging(seed=exp_config["seed"], logger_config={"type": JsonGzipLogger})
        .callbacks(dfaas_env.DFaaSCallbacks)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    )

    return config.build()


def main():
    epilog = """Some command line options can override the configuration of
    --exp-config option, if provided."""
    description = "Run a training experiment with the DFaaS environment."

    parser = argparse.ArgumentParser(prog="dfaas_train", description=description, epilog=epilog)

    parser.add_argument(dest="suffix", help="A string to append to experiment directory name")
    parser.add_argument("--exp-config", type=Path, help="Override default experiment config (YAML file)")
    parser.add_argument("--env-config", type=Path, help="Override default environment config (YAML file)")
    parser.add_argument("--runners", type=int, help="Number of parallel runners to play episodes")
    parser.add_argument("--seed", type=int, help="Seed of the experiment")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after one training iteration (useful for debugging purposes). Default is False",
    )

    args = parser.parse_args()

    # Read configuration files as dicts here, before calling run_experiment.
    if args.exp_config is not None:
        exp_config = dfaas_utils.yaml_to_dict(args.exp_config)
    else:
        exp_config = {}

    if args.env_config is not None:
        env_config = dfaas_utils.yaml_to_dict(args.env_config)
    else:
        env_config = {}

    # Pass config dicts directly to run_experiment instead of file paths.
    run_experiment(
        suffix=args.suffix,
        exp_config=exp_config,
        env_config=env_config,
        runners=args.runners,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
