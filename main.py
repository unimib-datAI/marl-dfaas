import sys
import json

# I need to log stdout and stderr to a log file. For this reason, I need to
# replace sys.stdout and sys.stderr with a custom wrapper as quickly as
# possible. This wrapper will be used transparently by Ray, RL4CC and my code.
#
# The only difference is that after the experiment is run, I can set the output
# path for the log, inside the "run()" function.
#
# This is a dirty hacky.
import utils
sys.stdout = utils.OutputDuplication()
sys.stderr = sys.stdout

from RL4CC.experiments.train import TrainingExperiment
from RL4CC.algorithms.algorithm import Algorithm
from RL4CC.utilities.postprocessing import evaluate_policy
from RL4CC.utilities.logger import Logger
from RL4CC.utilities.common import load_config_file, write_config_file

import traffic_env

# Logger used in this file.
logger = Logger(name="DFAAS", verbose=2)
logger_sep = "-" * 80

def main():
    experiment, policy = train()

    evaluation(experiment, policy)

def train():
    exp_config_path = "exp_config.json"
    exp_config = load_config_file(exp_config_path)
    env_config = load_config_file(exp_config["env_config_file"])

    logger.log(logger_sep)
    logger.log(f"START of training experiment")
    logger.log(f"Algorithm: {exp_config['algorithm']}")
    logger.log(f"Environment scenario: {env_config['scenario']}")
    logger.log(f"Config files relative paths:")
    logger.log(f"   exp_config: {exp_config_path}")
    logger.log(f"   ray_config: {exp_config['ray_config_file']}")
    logger.log(f"   env_config: {exp_config['env_config_file']}")
    logger.log(logger_sep)

    exp = TrainingExperiment(exp_config_path)
    policy = exp.run()

    # Set logging output file.
    sys.stdout.set_logfile(exp.logdir + "/main.log")

    logger.log(logger_sep)
    logger.log(f"END of training experiment")
    logger.log(logger_sep)

    return exp, policy

def get_eval_config(exp_config):
    """Returns a dictionary representing the evaluation configuration.

    The evaluation config is extracted and built from the given experiment
    config.

    The returned dictionary has the following keys:

        * "num_episodes_for_scenario" of type int (default is 1),
        * "allow_exploration" of type bool (default is False).
    """
    # The evaluation config dictionary is written directly to the given
    # experiment config. RL4CC ignores any unknown keys in the config. This step
    # is also done after RL4CC has trained the model.
    if "evaluation" not in exp_config.keys():
        exp_config["evaluation"] = {}

    eval_config = exp_config["evaluation"]

    num_eps_key = "num_episodes_for_scenario"
    if num_eps_key not in eval_config.keys():
        eval_config[num_eps_key] = 1
    elif (t := type(eval_config[num_eps_key])) is not int:
        logger.err(f"{num_eps_key!r} must be of type int, is {t.__name__}")
        sys.exit(1)

    explore_key = "allow_exploration"
    if explore_key not in eval_config.keys():
        eval_config[explore_key] = False
    elif (t := type(eval_config[explore_key])) is not bool:
        logger.err(f"{explore_key!r} must be of type bool, is {t.__name__}")
        sys.exit(1)

    return eval_config


def evaluation(experiment, policy):
    config = get_eval_config(experiment.exp_config)

    episodes = config["num_episodes_for_scenario"]
    allow_exploration = config["allow_exploration"]

    logger.log(logger_sep)
    logger.log(f"START of evaluation")
    logger.log(f"  Episodes for scenario: {episodes}")
    logger.log(f"  Allow exploration: {allow_exploration}")
    logger.log(logger_sep)

    # Results dictionary. "evaluations" contains the results for each scenario
    # evaluation.
    results = {
        "train_scenario": experiment.env_config["scenario"],
        "num_episodes_for_scenario": episodes,
        "allow_exploration": allow_exploration,
        "evaluations": {}
    }

    # Scenarios to test.
    for scenario in ["scenario" + str(i+1) for i in range(3)]:
        # Get original env_config from the experiment.
        env_config_path = experiment.exp_config["env_config_file"]
        env_config = load_config_file(env_config_path)

        # Set the scenario the policy will be evaluated with.
        env_config["scenario"] = scenario
        env = traffic_env.TrafficManagementEnv(env_config)

        # Evaluate the model with the specified scenario.
        logger.log(f"Evaluating policy with scenario {scenario!r}")
        result = evaluate_policy(policy=policy,
                                 env=env,
                                 num_eval_episodes=episodes,
                                 explore=allow_exploration)

        results["evaluations"][scenario] = result

    # Save results as JSON file to disk.
    dump = json.dumps(results, cls=utils.NumpyEncoder)
    path = write_config_file(dump, experiment.logdir, "evaluations_scenarios.json")

    logger.log(logger_sep)
    logger.log(f"END of evaluation")
    logger.log(f"   Raw results saved in {path!r}")
    logger.log(logger_sep)

def from_checkpoint():
    print("--- From checkpoint!")

    logdir = "/home/emanuele/ray_results/PPO_TrafficManagementEnv_2024-05-29_11-54-17rq9ys4jm"

    algo = Algorithm(algo_name="PPO", checkpoint_path=logdir + "/checkpoints/1")
    print(algo)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "checkpoint":
        from_checkpoint()
        exit(0)

    main()
