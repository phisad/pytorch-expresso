import os
from torchexpresso.runners import Trainer

from torchexpresso.configs import ExperimentConfigLoader


def perform_training(experiment_name, dry_run):
    config = ExperimentConfigLoader("configs") \
        .with_experiment_params(dry_run=dry_run) \
        .with_placeholders(checkpoint_dir=os.getenv("CHECKPOINT_DIR")) \
        .load(experiment_name, comet_user=os.getenv("COMET_USER"))
    Trainer.from_config(config, "train", "dev").perform()


if __name__ == '__main__':
    perform_training("classify/classify-digits-cnn", dry_run=False)
