import os
from torchexpresso.runners import Trainer, Predictor

from torchexpresso.configs import ExperimentConfigLoader


def perform_prediction():
    config = ExperimentConfigLoader("configs") \
        .with_experiment_params(dry_run=True) \
        .load("classify-digits", comet_user=os.getenv("COMET_USER"))
    Predictor.from_config(config, "test", os.getenv("MODEL_PATH")).perform()


if __name__ == '__main__':
    perform_prediction()
