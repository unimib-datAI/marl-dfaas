# This is an experiment with the Simplex action space with Ray.
#
# Run this script from the project root directory, not inside this directory!
# Otherwise the RL4CC module won't be loaded.
import sys
import os

# Add the current directory (where Python is called) to sys.path. This assumes
# this script is called in the project root directory, not inside the directory
# where the script is.
sys.path.append(os.getcwd())

from RL4CC.experiments.train import TrainingExperiment
from RL4CC.algorithms.algorithm import Algorithm

from simplex_env import SimplexTestEnv

def main():
    exp = TrainingExperiment("test-simplex/exp_config.json")
    policy = exp.run()

def from_checkpoint(checkpoint_path):
    checkpoint_path = checkpoint_path + "/checkpoints/1"

    algo = Algorithm(algo_name="PPO", checkpoint_path=checkpoint_path)

    print(algo)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        from_checkpoint(sys.argv[1])
        exit(0)

    main()
