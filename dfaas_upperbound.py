"""This module contains the SingleDFaaS environment, a simplified environment of
DFaaS where there is only a single large node that has the same resources as all
nodes in the corresponding DFaaS environment.

When run as main, this module performs a "training" experiment using the APL
heuristic. This is done to compare the real experiment with the theoretical
upper bound."""

from pathlib import Path
import shutil
from datetime import datetime
import logging
import argparse
from itertools import zip_longest
from collections import deque, defaultdict

import gymnasium as gym
import numpy as np

from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env

import dfaas_env
import dfaas_utils
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


class SingleDFaaS(gym.Env):
    """SingleDFaaS is a variant of the DFaaS environment in which there is a
    single centralized node that can only handle requests locally.

    For a given DFaaS environment, there is a corresponding SingleDFaaS
    environment, where the single node has the same resources (queue capacity,
    CPU, RAM...) as the sum of each node in the latter environment.

    The environment can be built with the same environment configuration for
    DFaaS.

    The DFaaS callback (DFaaSCallback) is compatible with SingleDFaaS."""

    def __init__(self, config={}):
        # Keep and use the reference environment to minimize code duplication.
        self.ref_env = dfaas_env.DFaaS(config=config)

        self.agents = ["node"]

        self.max_steps = self.ref_env.max_steps

        # Define a dummy action space: the agent can only process requests
        # locally.
        self.action_space = gym.spaces.Discrete(2)

        self.queue_capacity = self.ref_env.queue_capacity * len(self.ref_env.agents)

        self.input_requests_max = 150 * len(self.ref_env.agents)

        # The first two observations refer to the current step's starting point,
        # the next to the previous step (historical data).
        self.observation_space = gym.spaces.Dict(
            {
                # Number of input requests to process for a single step.
                "input_requests": gym.spaces.Box(
                    low=0, high=self.input_requests_max, dtype=np.int32
                ),
                # Queue current size.
                "queue_size": gym.spaces.Box(
                    low=0,
                    high=self.queue_capacity,
                    dtype=np.int32,
                ),
                # Local requests in the previous step.
                "prev_local_requests": gym.spaces.Box(
                    low=0, high=self.input_requests_max, dtype=np.int32
                ),
                # Local requests but rejected in the previosu step.
                # Note prev_local_rejects <= prev_local_requests.
                "prev_local_rejects": gym.spaces.Box(
                    low=0, high=self.input_requests_max, dtype=np.int32
                ),
            }
        )

        super().__init__()

    def get_config(self):
        """Returns a dictionary with the current configuration of the
        environment."""
        config = {
            "queue_capacity": self.queue_capacity,
            "max_steps": self.max_steps,
            "input_requests_type": self.ref_env.input_requests_type,
            "evaluation": self.ref_env.evaluation,
            "network": ["node"],
        }

        return config

    def reset(self, *, seed=None, options=None):
        self.current_step = 0

        # Reset the reference env to reuse the input request generation code.
        self.ref_env.reset(seed=seed, options=options)
        self.seed = self.ref_env.seed

        # Get the input requests of all agents in the reference environment and
        # sum by column (each column is a single step).
        input_requests_matrix = np.array(list(self.ref_env.input_requests.values()))
        self.input_requests = np.sum(input_requests_matrix, axis=0)

        # Copy some info from the reference env to be compatible with
        # DFaaSCallbacks.
        self.input_requests_type = self.ref_env.input_requests_type
        if self.input_requests_type == "real":
            self.input_requests_hashes = self.ref_env.input_requests_hashes

        self.queue = deque()

        def info_init_key():
            """Helper function to automatically initialize the keys in the info
            dictionary."""
            return [0 for _ in range(self.max_steps)]

        # See the DFaaS environment reset() method.
        self.info = defaultdict(info_init_key)

        obs = self._build_observation()

        return obs, {}

    def step(self, action_dict):
        # Ignore the action because the single node can only process input
        # requests locally.
        input_requests = self.input_requests[self.current_step]

        # Log the number of requests in the info dict. Keep the log structure as
        # DFaaS for better compatibility.
        self.info["action_local"][self.current_step] = input_requests
        self.info["action_forward"][self.current_step] = 0
        self.info["action_reject"][self.current_step] = 0

        # 2. Manage the workload.
        additional_rejects = self._manage_workload()

        # 3. Calculate the reward.
        #
        # WARNING: Since there is no action, the reward is a dummy value.
        reward = 1

        # Go to the next step.
        self.current_step += 1

        # 4. Update environment state.
        if self.current_step < self.max_steps:
            obs = self._build_observation()
        else:
            # Return a dummy observation because this is the last step.
            obs = self.observation_space.sample()

        # 5. Prepare return values.

        terminated = self.current_step == self.max_steps
        truncated = False  # Not used.

        # Postprocess the info dictionary by converting all NumPy arrays to
        # Python types. Note that this is done on the current and previous step
        # to avoid copying this code into the reset() method.
        #
        # TODO: This is not very efficient, but it works.
        for key in self.info:
            # Warning: this is also executed at the end of the episode!
            if self.current_step < self.max_steps:
                value = self.info[key][self.current_step]
                if isinstance(value, np.ndarray):
                    self.info[key][self.current_step] = value.item()

            value = self.info[key][self.current_step - 1]
            if isinstance(value, np.ndarray):
                self.info[key][self.current_step - 1] = value.item()

        return obs, reward, terminated, truncated, {}

    def _build_observation(self):
        """Builds and returns the observation for the current step."""
        assert self.current_step < self.max_steps

        def update_info(obs):
            """Helper function that populates the info dictionary for the
            current step only for the observation keys."""
            for key in obs.keys():
                self.info[f"observation_{key}"][self.current_step] = obs[key]

        # Initialize the observation dictionary. Note this is a single agent
        # environment.
        obs = {}

        # Special case: there is no data from the previous step at the start.
        if self.current_step == 0:
            input_requests = self.input_requests[self.current_step]
            obs = {
                "queue_size": np.array([0], dtype=np.int32),
                "input_requests": np.array([input_requests], dtype=np.int32),
                "prev_local_requests": np.array([0], dtype=np.int32),
                "prev_local_rejects": np.array([0], dtype=np.int32),
            }

            update_info(obs)
            return obs

        queue_size = self.info["queue_size"][self.current_step - 1]
        input_requests = self.input_requests[self.current_step]
        prev_local_reqs = self.info["action_local"][self.current_step - 1]
        prev_local_rejects = self.info["local_rejects_queue_full"][
            self.current_step - 1
        ]

        obs["queue_size"] = np.array([queue_size], dtype=np.int32)
        obs["input_requests"] = np.array([input_requests], dtype=np.int32)
        obs["prev_local_requests"] = np.array([prev_local_reqs], dtype=np.int32)
        obs["prev_local_rejects"] = np.array([prev_local_rejects], dtype=np.int32)

        update_info(obs)
        return obs

    def _manage_workload(self):
        """Manages the workload for the agent in the current step. It is based
        on the same method of the DFaaS environment.

        Returns the number of rejected local requests since the queue is
        full."""
        # CPU shares and RAM available for the node in a single step.
        cpu_capacity = 1000 * len(self.ref_env.agents)
        ram_capacity = 8000 * len(self.ref_env.agents)

        # There is no action: all input requests are processed locally.
        local = self.input_requests[self.current_step]

        def process_request(request):
            """Process the specified request. Returns True if the request has
            been processed in the current step, otherwise returns False."""
            nonlocal cpu_capacity, ram_capacity
            if cpu_capacity >= request.cpu_shares and ram_capacity >= request.ram_mb:
                cpu_capacity -= request.cpu_shares
                ram_capacity -= request.ram_mb
                return True  # Simulate the processing of the request.

            return False

        def append_queue(request):
            """Appends the specified requests to the queue. Returns True if the
            request was appended, otherwise False (the queue is full)."""
            if len(self.queue) < self.queue_capacity:
                self.queue.append(request)
                return True

            return False  # No available space.

        # 1. First, process the requests in the queue from the previous step.
        not_processed = deque()
        for request in self.queue:
            if process_request(request):
                self.info["processed_local"][self.current_step] += 1
            else:
                not_processed.append(request)

        self.queue = not_processed
        self.info["queue_size_pre_incoming_local"][self.current_step] = len(
            not_processed
        )

        # 2. Handle incoming local requests.
        #
        # Since this is a single giant node, we must first collect the sample of
        # all incoming requests and process them one by one, cycling through all
        # agents. This is necessary to keep the order.
        requests = []
        for agent in self.ref_env.agents:
            agent_reqs = dfaas_env._sample_workload(
                self.ref_env.input_requests[agent][self.current_step], self.ref_env.rng
            )

            requests.append(agent_reqs)

        # Note that a list may be shorter than another, this is why zip_longest.
        rejects = 0
        for batch in zip_longest(*requests):
            # Each request is from a different agent.
            for request in batch:
                if not request:
                    # We cycle between agents, the number of input requests can
                    # be different.
                    continue

                # Try to process incoming requests only when the queue is empty.
                if len(self.queue) == 0 and process_request(request):
                    self.info["processed_local"][self.current_step] += 1
                    continue

                # Queue is not empty or system does not have enough resources, try
                # to add the request to the queue.
                if append_queue(request):
                    continue

                # Insufficient requests and full queue: the only option left is to
                # reject.
                rejects += 1
                self.info["local_rejects_queue_full"][self.current_step] += 1

                self.info["queue_size"][self.current_step] = len(self.queue)

        return rejects


# Register the SingleDFaaS environment to be recognized by Ray.
dfaas_env.register(SingleDFaaS)


def main():
    """Execute the training experiment. Mainly copied from dfaas_train.py."""
    # TODO: Merge with dfaas_train.py
    parser = argparse.ArgumentParser(
        prog="dfaas_upperbound", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="suffix", help="A string to append to experiment directory"
    )
    parser.add_argument("--env-config", help="Environment config file")
    parser.add_argument(
        "--iterations",
        default=500,
        type=int,
        help="Number of iterations to run (non-negative integer)",
    )
    parser.add_argument(
        "--runners",
        default=5,
        type=int,
        help="Number of runners for collecting experencies in each iteration",
    )
    parser.add_argument(
        "--skip-evaluation",
        default=False,
        action="store_true",
        help="Skip final evaluation",
    )
    parser.add_argument("--seed", default=42, type=int, help="Seed of the experiment")

    args = parser.parse_args()

    if args.seed < 0:
        print("seed must be a non-negative number")
        exit(1)

    # TODO: Make this configurable from command line!
    exp_config = {
        "algorithm": dfaas_apl.APL.__name__,
        "seed": args.seed,
        "max_iterations": args.iterations,
        "env": SingleDFaaS.__name__,
        "final_evaluation": not args.skip_evaluation,
        "runners": args.runners,
    }
    logger.info(f"Experiment configuration = {exp_config}")

    # Environment configuration.
    if args.env_config is not None:
        env_config = dfaas_utils.json_to_dict(args.env_config)
    else:
        env_config = {}

    # Create a dummy environment, used as reference.
    dummy_env = SingleDFaaS(config=env_config)
    logger.info(f"Environment configuration = {dummy_env.get_config()}")

    # For the evaluation phase at the end, the env_config is different than the
    # training one.
    env_eval_config = env_config.copy()
    env_eval_config["evaluation"] = True

    assert dummy_env.max_steps == 288, "Only 288 steps supported"

    experiment = build_algorithm(
        runners=args.runners,
        dummy_env=dummy_env,
        env_config=env_config,
        env_eval_config=env_eval_config,
        exp_config=exp_config,
    )

    # Get the experiment directory to save other files.
    logdir = Path(experiment.logdir).resolve()
    logger.info(
        f"{SingleDFaaS.__name__} experiment directory created at {logdir.as_posix()!r}"
    )
    # This will be used after the evaluation.
    start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    max_iterations = exp_config["max_iterations"]
    for iteration in range(max_iterations):
        percentual = (iteration + 1) / max_iterations
        logger.info(f"Iteration {iteration + 1}/{max_iterations} {percentual:.0%}")
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
    exp_name = f"{SingleDFaaS.__name__}_{start}_{dfaas_apl.APL.__name__}_{args.suffix}"
    result_dir = Path.cwd() / "results" / exp_name
    shutil.move(logdir, result_dir.resolve())
    logger.info(
        f"{SingleDFaaS.__name__} experiment results moved to {result_dir.as_posix()!r}"
    )


def build_algorithm(**kwargs):
    runners = kwargs["runners"]
    dummy_env = kwargs["dummy_env"]
    env_config = kwargs["env_config"]
    env_eval_config = kwargs["env_eval_config"]
    exp_config = kwargs["exp_config"]

    # Each runner play three episodes, so in each iteration the episodes playied
    # depends on the number of runners (min 1, usually 5).
    episodes_iter = 3 * (runners if runners > 0 else 1)

    # The train batch size must match exactly the steps collected to avoid
    # truncated episodes in the logs.
    train_batch_size = dummy_env.max_steps * episodes_iter

    config = (
        dfaas_apl.APLConfig()
        # By default RLlib uses the new API stack, but I use the old one.
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        # For each iteration, store only the episodes calculated in that
        # iteration in the log result.
        .reporting(metrics_num_episodes_for_smoothing=episodes_iter)
        .environment(env=SingleDFaaS.__name__, env_config=env_config)
        .training(train_batch_size=train_batch_size)
        .env_runners(num_env_runners=runners)
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=50,
            evaluation_num_env_runners=1,
            evaluation_config={"env_config": env_eval_config},
        )
        .debugging(seed=exp_config["seed"])
        .callbacks(dfaas_env.DFaaSCallbacks)
    )

    return config.build()


if __name__ == "__main__":
    main()
