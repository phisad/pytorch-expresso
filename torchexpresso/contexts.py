import comet_ml
import tempfile
import torch
from torch import cuda
from torch.utils import data
import os
import logging
from inspect import signature

from torchexpresso.callbacks.metrics import AverageLossMetric

from torchexpresso.callbacks import CallbackRegistry

from torchexpresso import steps
from torchexpresso.savers import SaverRegistry, load_checkpoint_from_path

logger = logging.getLogger(__file__)

PARAM_DRY_RUN = "dry_run"
"""Optional experiment param flag to run only one step for a single episode """

PARAM_CPU_ONLY = "cpu_only"
"""Optional experiment param flag to force to use the CPU """

PARAM_RESUME = "resume"
"""Optional experiment param flag to resume training from the latest checkpoint """


def is_dryrun(exp_params):
    if PARAM_DRY_RUN in exp_params:
        return exp_params[PARAM_DRY_RUN]
    return False


class ContextLoader:

    @staticmethod
    def load_module_from_config(module_config):
        if "kwargs" in module_config:
            return ContextLoader.load_module_dynamically(module_config["package"], module_config["class"],
                                                         module_config["kwargs"])
        return ContextLoader.load_module_dynamically(module_config["package"], module_config["class"])

    @staticmethod
    def load_module_dynamically(python_package, python_class, cls_kwargs=None):
        cls_package = __import__(python_package, fromlist=python_class)
        cls = getattr(cls_package, python_class)
        if cls_kwargs is None:
            return cls()
        return cls(**cls_kwargs)

    @staticmethod
    def load_optimizer_from_config(module_config, model):
        if "kwargs" in module_config:
            return ContextLoader.load_optimizer_dynamically(module_config["package"], module_config["class"], model,
                                                            module_config["kwargs"])
        return ContextLoader.load_optimizer_dynamically(module_config["package"], module_config["class"], model)

    @staticmethod
    def load_optimizer_dynamically(python_package, python_class, model, cls_kwargs=None):
        cls_package = __import__(python_package, fromlist=python_class)
        cls = getattr(cls_package, python_class)
        if cls_kwargs is None:
            return cls(model.parameters())
        return cls(model.parameters(), **cls_kwargs)

    @staticmethod
    def load_device_from_config(experiment_config):
        cpu_only = False
        if PARAM_CPU_ONLY in experiment_config["params"]:
            cpu_only = experiment_config["params"][PARAM_CPU_ONLY]
        return ContextLoader.load_device(cpu_only)

    @staticmethod
    def load_device(force_cpu=False):
        if force_cpu:
            device_name = "cpu"
        else:
            device_name = "cuda:0" if cuda.is_available() else "cpu"
        return torch.device(device_name)

    @staticmethod
    def load_callbacks_from_config(exp_config, partial_context):
        callbacks = CallbackRegistry()
        # Note: The ordering actually matters!
        if "callbacks" in exp_config:
            clb_configs = exp_config["callbacks"]
            for clb_config in clb_configs:
                # Special handler for callbacks merger
                if "kwargs" in clb_config:
                    clb_kwargs = clb_config["kwargs"]
                    if "metrics" in clb_kwargs:
                        # We need to get the metrics from the registry
                        ref_metrics = [callbacks[ref] for ref in clb_kwargs["metrics"]]
                        # We replace the strings with the instances (now it is loadable dynamically)
                        clb_kwargs["metrics"] = ref_metrics
                clb = ContextLoader.load_callback(clb_config, partial_context)
                callbacks[clb.name] = clb  # Every callback needs a name
        else:
            # TODO do we want to set default callbacks?
            loss_default = AverageLossMetric(partial_context["comet"])
            callbacks[loss_default.name] = loss_default
        return callbacks

    @staticmethod
    def load_callback(clb_config, partial_context):
        if "kwargs" in clb_config:
            return ContextLoader.load_callback_dynamically(clb_config["package"], clb_config["class"], partial_context,
                                                           clb_config["kwargs"])
        return ContextLoader.load_callback_dynamically(clb_config["package"], clb_config["class"], partial_context)

    @staticmethod
    def load_callback_dynamically(python_package, python_class, partial_context, clb_kwargs=None):
        clb_package = __import__(python_package, fromlist=python_class)
        clb_class = getattr(clb_package, python_class)

        if clb_kwargs is None:
            clb_kwargs = dict()

        sig = signature(clb_class)
        for p in sig.parameters.values():
            if p.name in ["comet", "experiment"]:
                clb_kwargs[p.name] = partial_context["comet"]
            if p.name in ["model"]:
                clb_kwargs[p.name] = partial_context["model"]

        return clb_class(**clb_kwargs)

    @staticmethod
    def load_savers_from_config(exp_config):
        savers = SaverRegistry()
        # Note: The ordering actually matters!
        if "savers" in exp_config:
            svr_configs = exp_config["savers"]
            for srv_config in svr_configs:
                svr = ContextLoader.load_saver(srv_config, exp_config["model"], exp_config["task"])
                savers[svr.name] = svr  # Every callback needs a name
        else:
            # TODO do we want to set default savers?
            ...
        return savers

    @staticmethod
    def load_saver(srv_config, model_config, task_config):
        if "kwargs" in srv_config:
            return ContextLoader.load_saver_dynamically(srv_config["package"], srv_config["class"],
                                                        model_config, task_config, srv_config["kwargs"])
        return ContextLoader.load_saver_dynamically(srv_config["package"], srv_config["class"],
                                                    model_config, task_config)

    @staticmethod
    def load_saver_dynamically(python_package, python_class, model_config, task_config, svr_kwargs=None):
        svr_package = __import__(python_package, fromlist=python_class)
        svr_class = getattr(svr_package, python_class)
        if svr_kwargs is None:
            return svr_class(model_config, task_config)
        return svr_class(model_config, task_config, **svr_kwargs)

    @staticmethod
    def load_model_from_config(model_config, task):
        model_params = None
        if "params" in model_config:
            model_params = model_config["params"]
        return ContextLoader.load_model_dynamically(model_config["package"], model_config["class"],
                                                    model_params, task)

    @staticmethod
    def load_model_dynamically(python_package, python_class, model_params, task):
        model_package = __import__(python_package, fromlist=python_class)
        model_class = getattr(model_package, python_class)
        if model_params is None:
            return model_class(task)
        return model_class(task, model_params)

    @staticmethod
    def load_providers_from_config(experiment_config, split_names: list, device):
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
        env_params = None
        if "params" in env_config:
            env_params = env_config["params"]
        return ContextLoader.load_env_dynamically(env_config["package"], env_config["class"],
                                                  env_params, task, split_name, device)

    @staticmethod
    def load_env_dynamically(python_package, python_class, env_params, task, split_name, device):
        env_package = __import__(python_package, fromlist=python_class)
        env_class = getattr(env_package, python_class)
        return env_class(task, env_params, split_name, device)

    @staticmethod
    def load_dataset_from_config(dataset_config, task, split_name, device):
        ds_params = None
        if "params" in dataset_config:
            ds_params = dataset_config["params"]
        return ContextLoader.load_dataset_dynamically(dataset_config["package"], dataset_config["class"],
                                                      ds_params, task, split_name, device)

    @staticmethod
    def load_dataset_dynamically(python_package, python_class, dataset_params, task, split_name, device):
        dataset_package = __import__(python_package, fromlist=python_class)
        dataset_class = getattr(dataset_package, python_class)
        return dataset_class(task, split_name, dataset_params, device)

    @staticmethod
    def load_cometml_experiment(comet_config, experiment_name):
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
        return experiment


def collate_variable_sequences(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    return x, y  # returning a pair of tensor-lists


def log_config(comet, experiment_config):
    def apply_prefix(params_dict, prefix):
        return dict([("%s-%s" % (prefix, name), v) for name, v in params_dict.items()])

    if "params" in experiment_config:
        comet.log_parameters(apply_prefix(experiment_config["params"], "exp"))
    if "model" in experiment_config:
        model_config = experiment_config["model"]
        comet.log_parameter("model-class", "%s.%s" % (model_config["package"], model_config["class"]))
        if "params" in model_config:
            comet.log_parameters(apply_prefix(model_config["params"], "model"))
    if "dataset" in experiment_config:
        dataset_config = experiment_config["dataset"]
        comet.log_parameter("ds-class", "%s.%s" % (dataset_config["package"], dataset_config["class"]))
        if "params" in dataset_config:
            comet.log_parameters(apply_prefix(dataset_config["params"], "ds"))
    if "task" in experiment_config:
        comet.log_parameters(apply_prefix(experiment_config["task"], "task"))


def log_checkpoint(comet, checkpoint):
    if comet:
        for entry_name in checkpoint:
            if entry_name.startswith("cp-"):  # ckpt-param
                comet.log_other(entry_name, checkpoint[entry_name])


class ExperimentContext:

    @classmethod
    def from_config(cls, experiment_config, split_names: list):
        """ Create a experiment context from the config"""
        partial_context = dict()
        partial_context["config"] = experiment_config

        """ Load and setup the cometml experiment """
        comet = ContextLoader.load_cometml_experiment(experiment_config["cometml"], experiment_config["name"])
        if "tags" in experiment_config:
            comet.add_tags(experiment_config["tags"])
        log_config(comet, experiment_config)
        partial_context["comet"] = comet

        """ Load and setup the device """
        device = ContextLoader.load_device_from_config(experiment_config)
        partial_context["device"] = device

        """ Load the data providers """
        providers = ContextLoader.load_providers_from_config(experiment_config, split_names, device)
        partial_context["providers"] = providers

        return cls(partial_context)

    def __init__(self, experiment_context):
        self.holder = experiment_context

    def __getitem__(self, item):
        return self.holder[item]

    def __setitem__(self, key, value):
        self.holder[key] = value


class TrainingContext:

    @classmethod
    def from_config(cls, experiment_config, split_names):
        """ Create a training context from the config"""

        """ Load experiment context """
        partial_context = ExperimentContext.from_config(experiment_config, split_names)

        """ Load and setup model """
        epoch_start = 1
        model = ContextLoader.load_model_from_config(experiment_config["model"], experiment_config["task"])

        """ Handle optional resume """
        is_resume = PARAM_RESUME in experiment_config["params"] and experiment_config["params"][PARAM_RESUME]
        if is_resume:
            """ Load the checkpoint """
            ckpt = load_checkpoint_from_path(experiment_config["params"]["resume_checkpoint_path"])
            log_checkpoint(partial_context["comet"], ckpt)

            # Note: In constrast to predict, we use the exp-model and exp-task
            model.load_state_dict(ckpt['state_dict'], strict=False)

            epoch_start = ckpt["cp-epoch"] + 1
            print("Resume training from epoch: {:d}".format(epoch_start))
        model.to(partial_context["device"])
        partial_context["epoch_start"] = epoch_start
        partial_context["model"] = model

        # Load optimizer only now to guarantee that the parameters are on the same device
        if "optim_fn" in experiment_config["params"]:
            optimizer = ContextLoader.load_optimizer_from_config(experiment_config["params"]["optim_fn"], model)
        else:
            optimizer = torch.optim.Adam(model.parameters())

        if is_resume:
            optimizer.load_state_dict(ckpt["optimizer"])
        partial_context["optimizer"] = optimizer

        """ Load the savers """
        savers = ContextLoader.load_savers_from_config(experiment_config)
        partial_context["savers"] = savers

        """ Load and setup the loss function """
        if "loss_fn" in experiment_config["params"]:
            loss_fn = ContextLoader.load_module_from_config(experiment_config["params"]["loss_fn"])
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        partial_context["loss_fn"] = loss_fn

        """ Load and setup the step function """
        if "step_fn" in experiment_config["params"]:
            step_fn = ContextLoader.load_module_from_config(experiment_config["params"]["step_fn"])
        else:
            step_fn = steps.TrainingStep()
        partial_context["step_fn"] = step_fn

        """ Load the callbacks (only after the model, b.c. there are callbacks that require the model) """
        callbacks = ContextLoader.load_callbacks_from_config(experiment_config, partial_context)
        partial_context["callbacks"] = callbacks

        return cls(partial_context)

    def __init__(self, experiment_context):
        self.holder = dict()
        self.holder = {**self.holder, **experiment_context.holder}

    def __getitem__(self, item):
        return self.holder[item]

    def is_dryrun(self):
        return is_dryrun(self.holder["config"]["params"])


class PredictionContext:

    @classmethod
    def from_config(cls, experiment_config: dict, split_names: list, model_path: str):
        """ Create a prediction context from the config"""
        if model_path is None:
            raise Exception("Missing 'model_path' argument. Please provide a path to the model.")

        """ Load experiment context """
        partial_context = ExperimentContext.from_config(experiment_config, split_names)

        """ Load the checkpoint """
        ckpt = load_checkpoint_from_path(model_path)
        log_checkpoint(partial_context["comet"], ckpt)

        """ Load and setup the model from ckpt"""
        model = ContextLoader.load_model_from_config(ckpt["cp-model"], ckpt["cp-task"])
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.to(partial_context["device"])
        partial_context["model"] = model

        """ Load the callbacks (only after the model, b.c. there are callbacks that require the model) """
        callbacks = ContextLoader.load_callbacks_from_config(experiment_config, partial_context)
        partial_context["callbacks"] = callbacks

        return cls(partial_context)

    def __init__(self, experiment_context, model, callbacks):
        self.holder = dict()
        self.holder = {**self.holder, **experiment_context.holder}

    def __getitem__(self, item):
        return self.holder[item]

    def is_dryrun(self):
        return is_dryrun(self.holder["config"]["params"])


class ProcessorContext:

    @classmethod
    def from_config(cls, experiment_config, split_names):
        """ Create a processor context from the config"""

        """ Load experiment context """
        experiment_context = ExperimentContext.from_config(experiment_config, split_names)

        return cls(experiment_context)

    def __init__(self, experiment_context):
        self.holder = dict()
        self.holder = {**self.holder, **experiment_context.holder}

    def __getitem__(self, item):
        return self.holder[item]

    def is_dryrun(self):
        return is_dryrun(self.holder["config"]["params"])
