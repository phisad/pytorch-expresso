import comet_ml
import tempfile
import torch
from torch import cuda
from torch.utils import data
import os
import logging

from torchexpresso import savers

logger = logging.getLogger(__file__)


class ContextLoader:

    @staticmethod
    def load_device(force_cpu=False):
        if force_cpu:
            device_name = "cpu"
        else:
            device_name = "cuda:0" if cuda.is_available() else "cpu"
        return torch.device(device_name)

    @staticmethod
    def load_loss_fn(loss_config):
        if "kwargs" in loss_config:
            return ContextLoader.load_loss_fn_dynamically(loss_config["package"], loss_config["class"],
                                                          loss_config["kwargs"])
        return ContextLoader.load_loss_fn_dynamically(loss_config["package"], loss_config["class"])

    @staticmethod
    def load_loss_fn_dynamically(python_package, python_class, loss_kwargs=None):
        loss_package = __import__(python_package, fromlist=python_class)
        loss_class = getattr(loss_package, python_class)
        if loss_kwargs is None:
            return loss_class()
        return loss_class(**loss_kwargs)

    @staticmethod
    def load_model_from_config(model_config, task):
        return ContextLoader.load_model_dynamically(model_config["package"], model_config["class"],
                                                    model_config["params"],
                                                    task)

    @staticmethod
    def load_model_dynamically(python_package, python_class, model_params, task):
        model_package = __import__(python_package, fromlist=python_class)
        model_class = getattr(model_package, python_class)
        return model_class(task, model_params)

    @staticmethod
    def load_providers_from_config(experiment_config, split_names, device):
        providers = dict()
        # Note: If there is an environment in the config, then we use that as the dataset 'split'
        if "env" in experiment_config:
            env = ContextLoader.load_env_from_config(experiment_config["env"],
                                                     experiment_config["task"], "env", device)
            # It seems that we cannot run two seperate envs for 'train' and 'dev'
            # causing "pyglet.gl.lib.GLException: b'invalid value'" so that we use the same env for both splits
        for split_name in split_names:
            split_or_env = split_name
            if "env" in experiment_config:
                experiment_config["dataset"]["params"]["split_name"] = split_name
                split_or_env = env
            dataset = ContextLoader.load_dataset_from_config(experiment_config["dataset"],
                                                             experiment_config["task"],
                                                             split_or_env, device)
            provider = data.dataloader.DataLoader(dataset,
                                                  batch_size=experiment_config["params"]["batch_size"],
                                                  shuffle=split_name == "train" and "env" not in experiment_config,
                                                  collate_fn=collate_variable_sequences)
            providers[split_name] = provider
        return providers

    @staticmethod
    def load_env_from_config(env_config, task, split_name, device):
        return ContextLoader.load_env_dynamically(env_config["package"], env_config["class"],
                                                  env_config["params"], task, split_name, device)

    @staticmethod
    def load_env_dynamically(python_package, python_class, params, task, split_name, device):
        env_package = __import__(python_package, fromlist=python_class)
        env_class = getattr(env_package, python_class)
        return env_class(task, params, split_name, device)

    @staticmethod
    def load_dataset_from_config(dataset_config, task, split_name, device):
        return ContextLoader.load_dataset_dynamically(dataset_config["package"], dataset_config["class"],
                                                      dataset_config["params"],
                                                      task, split_name, device)

    @staticmethod
    def load_dataset_dynamically(python_package, python_class, dataset_params, task, split_name, device):
        dataset_package = __import__(python_package, fromlist=python_class)
        dataset_class = getattr(dataset_package, python_class)
        return dataset_class(task, split_name, dataset_params, device)

    @staticmethod
    def load_cometml_experiment(comet_config, experiment_name, tags):
        # Optionals
        cometml_workspace = None
        cometml_project = None
        if "workspace" in comet_config:
            cometml_workspace = comet_config["workspace"]
        if "project_name" in comet_config:
            cometml_project = comet_config["project_name"]

        if comet_config["offline"]:
            # Optional offline directory
            offline_directory = None
            if "offline_directory" in comet_config:
                offline_directory = comet_config["offline_directory"]
            # Defaults to tmp-dir
            if offline_directory is None:
                offline_directory = os.path.join(tempfile.gettempdir(), "cometml")
            if not os.path.exists(offline_directory):
                os.makedirs(offline_directory)
            logger.info("Writing CometML experiments to %s", offline_directory)
            experiment = comet_ml.OfflineExperiment(workspace=cometml_workspace, project_name=cometml_project,
                                                    offline_directory=offline_directory)
        else:
            experiment = comet_ml.Experiment(workspace=cometml_workspace, project_name=cometml_project,
                                             api_key=comet_config["api_key"]
                                             )
        experiment.set_name(experiment_name)
        experiment.add_tags(tags)
        return experiment


def collate_variable_sequences(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    return x, y  # returning a pair of tensor-lists


def log_config(comet, experiment_config):
    def apply_prefix(params_dict, prefix):
        return dict([("%s-%s" % (prefix, name), v) for name, v in params_dict.items()])

    comet.log_parameters(apply_prefix(experiment_config["params"], "exp"))
    comet.log_parameters(apply_prefix(experiment_config["model"]["params"], "model"))
    comet.log_parameters(apply_prefix(experiment_config["dataset"]["params"], "ds"))
    comet.log_parameters(apply_prefix(experiment_config["task"], "task"))


def log_checkpoint(comet, checkpoint):
    if comet:
        comet.log_other("checkpoint_epoch", checkpoint["epoch"])
        comet.log_other("checkpoint_best_value", checkpoint["best_value"])
        comet.log_other("checkpoint_best_value_metric", checkpoint["best_value_metric"])
        comet.log_other("checkpoint_arch", checkpoint["arch"])


class TrainingContext:

    @classmethod
    def from_config(cls, experiment_config, split_names):
        """ Create a training context from the config"""

        """ Load and setup the cometml experiment """
        comet = ContextLoader.load_cometml_experiment(experiment_config["cometml"], experiment_config["name"],
                                                      experiment_config["tags"])
        log_config(comet, experiment_config)

        """ Load and setup model and optimizer"""
        epoch_start = 1
        device = ContextLoader.load_device(experiment_config["params"]["cpu_only"])
        model = ContextLoader.load_model_from_config(experiment_config["model"], experiment_config["task"])

        """ Handle optional resume """
        is_resume = "resume" in experiment_config["params"] and experiment_config["params"]["resume"]
        if is_resume:
            checkpoint = savers.CheckpointSaver.load_checkpoint(model,
                                                                experiment_config["params"]["checkpoint_dir"],
                                                                experiment_config["name"])
            log_checkpoint(comet, checkpoint)
            epoch_start = checkpoint["epoch"] + 1
            print("Resume training from epoch: {:d}".format(epoch_start))
        model.to(device)

        # Load optimizer only now to guarantee that the parameters are on the same device
        optimizer = torch.optim.Adam(model.parameters())
        if is_resume:
            optimizer.load_state_dict(checkpoint["optimizer"])

        """ Load and setup the loss function """
        if "loss_fn" in experiment_config["params"]:
            loss_fn = ContextLoader.load_loss_fn(experiment_config["params"]["loss_fn"])
        else:
            # Mask padding_value=0 for loss computation
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        """ Load the data providers """
        providers = ContextLoader.load_providers_from_config(experiment_config, split_names, device)

        return cls(experiment_config, comet, model, optimizer, loss_fn, providers, epoch_start, device)

    def __init__(self, config, comet, model, optimizer, loss_fn, providers, epoch_start, device):
        self.holder = dict()
        self.holder["config"] = config
        self.holder["comet"] = comet
        self.holder["model"] = model
        self.holder["optimizer"] = optimizer
        self.holder["loss_fn"] = loss_fn
        self.holder["providers"] = providers
        self.holder["epoch_start"] = epoch_start
        self.holder["device"] = device

    def __getitem__(self, item):
        return self.holder[item]


class PredictionContext:

    @classmethod
    def from_config(cls, experiment_config, split_names):
        """ Create a training context from the config"""

        """ Load and setup the cometml experiment """
        comet = ContextLoader.load_cometml_experiment(experiment_config["cometml"], experiment_config["name"],
                                                      experiment_config["tags"])
        log_config(comet, experiment_config)

        """ Load and setup the model """
        device = ContextLoader.load_device(experiment_config["params"]["cpu_only"])
        model = ContextLoader.load_model_from_config(experiment_config["model"], experiment_config["task"])
        if "checkpoint_dir" in experiment_config["params"]:
            checkpoint = savers.CheckpointSaver.load_checkpoint(model,
                                                                experiment_config["params"]["checkpoint_dir"],
                                                                experiment_config["name"])
            log_checkpoint(comet, checkpoint)
        model.to(device)

        """ Load the data providers """
        providers = ContextLoader.load_providers_from_config(experiment_config, split_names, device)
        return cls(experiment_config, comet, model, providers, device)

    def __init__(self, config, comet, model, providers, device):
        self.holder = dict()
        self.holder["config"] = config
        self.holder["comet"] = comet
        self.holder["model"] = model
        self.holder["providers"] = providers
        self.holder["device"] = device

    def __getitem__(self, item):
        return self.holder[item]
