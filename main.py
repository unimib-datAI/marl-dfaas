from RL4CC.experiments.train import TrainingExperiment
import environment

def main():
    exp = TrainingExperiment("exp_config.json")
    exp.run()


if __name__ == '__main__':
    main()
