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

def evaluation(experiment, policy):
    logger.log(logger_sep)
    logger.log(f"START of evaluation")
    logger.log(logger_sep)

    # Scenarios to tests.
    scenarios = ["scenario" + str(i+1) for i in range(3)]
    results = {
        "train_scenario": experiment.env_config["scenario"],
        "evaluations": {}
    }

    for scenario in scenarios:
        # Get original env_config from the experiment.
        env_config_path = experiment.exp_config["env_config_file"]
        env_config = load_config_file(env_config_path)

        # Set the scenario the policy will be evaluated with.
        env_config["scenario"] = scenario
        env = traffic_env.TrafficManagementEnv(env_config)

        logger.log(f"Evaluate policy with scenario {scenario!r}")

        results["evaluations"][scenario] = evaluate_policy(policy=policy, env=env)

    # Save results.
    dump = json.dumps(results, cls=utils.NumpyEncoder)
    path = write_config_file(dump, experiment.logdir, "scenarios_evaluations.json")

    logger.log(logger_sep)
    logger.log(f"END of evaluation")
    logger.log(f"Raw results saved in {path!r}")
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
