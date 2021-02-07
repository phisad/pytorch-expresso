from torchexpresso.runners import Trainer

from torchexpresso.configs import ExperimentConfigLoader


def perform_training():
    config = ExperimentConfigLoader("configs").load("classify-digits", comet_user="phisad")
    Trainer.from_config(config, "train", "dev").perform()


if __name__ == '__main__':
    perform_training()