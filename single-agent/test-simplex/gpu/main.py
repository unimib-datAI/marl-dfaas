# This is an experiment with the Simplex action space with Ray with GPU enabled.
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

from simplex_env import SimplexTestEnv

def main():
    exp = TrainingExperiment("test-simplex/gpu/exp_config.json")
    policy = exp.run()

    print(policy)

if __name__ == "__main__":
    main()
