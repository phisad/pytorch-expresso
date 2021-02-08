import torch
import logging
from torchexpresso import contexts
from torchexpresso.callbacks import CallbackRegistry
from torchexpresso.steps import TrainingStep, Step
from torchexpresso.savers import Saver, NoopSaver

logger = logging.getLogger(__file__)


class Trainer(object):

    @classmethod
    def from_config(cls, experiment_config: dict, train_split: str = "train", dev_split: str = "dev"):
        """
            @param experiment_config: a dictionary with all meta-information to perform the training
            @param train_split: Name of the dataset split on which to perform the training.
            @param dev_split: Name of the dataset split on which to perform the validation.
        """
        context = contexts.TrainingContext.from_config(experiment_config, [train_split, dev_split])
        return cls(context, train_split, dev_split)

    def __init__(self, context, train_split, dev_split):
        self.ctx = context
        self.train_split = train_split
        self.dev_split = dev_split

    def perform(self, callbacks: CallbackRegistry = None, saver: Saver = None, step: TrainingStep = None):
        logger.info("Perform training for the experiment '%s' ", self.ctx["config"]["name"])

        if callbacks is None:
            callbacks = self.ctx["callbacks"]

        if len(callbacks) == 0:
            logger.info("No callbacks registered!")
        else:
            logger.info("Using configured callbacks.")
            [logger.info("Detected callback: %s", clb) for clb in callbacks]

        if saver is None:
            saver = self.ctx["savers"]

        if len(saver) == 0:
            logger.info("No savers registered!")
        else:
            logger.info("Using configured savers.")
            [logger.info("Detected saver: %s", svr) for svr in saver]

        if step is None:
            step = self.ctx["step_fn"]
        epoch_start = self.ctx["epoch_start"]

        if self.ctx.is_dryrun():
            logger.info("Detected dry run mode. Performing a single episode step only.")

        """ Perform the training and validation """
        total_epochs = self.ctx["config"]["params"]["num_epochs"]
        for epoch in range(epoch_start, total_epochs + 1):
            self.__epoch_train(epoch, callbacks, step)
            self.__epoch_validate(epoch, callbacks, step)
            saver.on_epoch_end(self.ctx["model"], self.ctx["optimizer"], epoch, callbacks)
            if self.ctx.is_dryrun():
                break
        logger.info("Finished training for the experiment '%s' ", self.ctx["config"]["name"])

    def __epoch_validate(self, current_epoch, callbacks, step: TrainingStep):
        self.ctx["model"].eval()
        with self.ctx["comet"].validate(), torch.no_grad():
            self.__perform_steps("validate", current_epoch, callbacks, step)

    def __epoch_train(self, current_epoch, callbacks: CallbackRegistry, step: TrainingStep):
        self.ctx["model"].train()
        with self.ctx["comet"].train():
            self.__perform_steps("train", current_epoch, callbacks, step)

    def __perform_steps(self, current_phase, current_epoch, callbacks, step):
        callbacks.on_epoch_start(phase=current_phase, epoch=current_epoch)
        step.on_epoch_start(phase=current_phase, epoch=current_epoch)
        split_name = self.train_split if current_phase == "train" else self.dev_split
        provider = self.ctx["providers"][split_name]
        for current_step, (batch_inputs, batch_labels) in enumerate(provider):
            current_step = current_step + 1
            if current_phase == "train":
                self.ctx["optimizer"].zero_grad()
            batch_inputs = step.prepare(self.ctx["model"], batch_inputs, self.ctx["device"], step=current_step)
            batch_outputs, batch_inputs = step.forward(self.ctx["model"], batch_inputs, self.ctx["device"],
                                                       step=current_step)
            loss, batch_labels = step.loss(self.ctx["loss_fn"], batch_outputs, batch_labels, self.ctx["device"],
                                           step=current_step)
            if current_phase == "train":
                self.ctx["optimizer"].step()
            callbacks.on_step(inputs=batch_inputs, outputs=batch_outputs, labels=batch_labels,
                              mask=None, loss=loss, step=current_step)
            if self.ctx.is_dryrun():
                break
        callbacks.on_epoch_end(epoch=current_epoch)


class Predictor(object):

    @classmethod
    def from_config(cls, experiment_config, split_name, model_path: str):
        """
            @param experiment_config: a dictionary with all meta-information to perform the training
            @param split_name: Name of the dataset split on which to perform the prediction.
            @param model_path: a path to the model checkpoint e.g. /path/to/model/model.pth.tar
        """
        context = contexts.PredictionContext.from_config(experiment_config, [split_name], model_path)
        return cls(context, split_name)

    def __init__(self, context, split_name):
        self.ctx = context
        self.split_name = split_name

    def perform(self, callbacks: CallbackRegistry = None, step: Step = None):
        logger.info("Perform prediction for the experiment '%s' on '%s'", self.ctx["config"]["name"], self.split_name)

        if callbacks is None:
            callbacks = self.ctx["callbacks"]

        if len(callbacks) == 0:
            logger.info("No callbacks registered!")
        else:
            logger.info("Using configured callbacks.")
            [logger.info("Detected callback: %s", clb) for clb in callbacks]

        if step is None:
            step = Step()

        if self.ctx.is_dryrun():
            logger.info("Detected dry run mode. Performing a single step only.")

        """ Perform the prediction """
        self.ctx["model"].eval()
        with self.ctx["comet"].test(), torch.no_grad():
            callbacks.on_epoch_start(phase="test", epoch=None)
            provider = self.ctx["providers"][self.split_name]
            for current_step, (batch_inputs, batch_labels) in enumerate(provider):
                current_step = current_step + 1
                batch_inputs = step.prepare(self.ctx["model"], batch_inputs, self.ctx["device"], step=current_step)
                batch_outputs, batch_inputs = step.forward(self.ctx["model"], batch_inputs, self.ctx["device"],
                                                           step=current_step)
                callbacks.on_step(inputs=batch_inputs, outputs=batch_outputs, labels=batch_labels, mask=None, loss=None,
                                  step=current_step)
                if self.ctx.is_dryrun():
                    break
            callbacks.on_epoch_end(epoch=None)


class Processor(object):
    """
        Perform anything which does not involve a model. This can be used for pre-processing.

        Providers and callbacks are called as usualy.
    """

    @classmethod
    def from_config(cls, experiment_config, split_name):
        """
            @param experiment_config: a dictionary with all meta-information to perform the training
            @param split_name: Name of the dataset split on which to perform the process.
        """
        context = contexts.ProcessorContext.from_config(experiment_config, [split_name])
        return cls(context, split_name)

    def __init__(self, context, split_name):
        self.ctx = context
        self.split_name = split_name

    def perform(self, callbacks: CallbackRegistry = None):
        logger.info("Perform processing for the experiment '%s' on '%s'", self.ctx["config"]["name"], self.split_name)

        if callbacks is None:
            callbacks = self.ctx["callbacks"]

        if len(callbacks) == 0:
            logger.info("No callbacks registered!")
        else:
            logger.info("Using configured callbacks.")
            [logger.info("Detected callback: %s", clb) for clb in callbacks]

        epoch_start = 1
        total_epochs = 1
        if "params" in self.ctx["config"]:
            exp_params = self.ctx["config"]["params"]
            if "epoch_start" in exp_params:
                epoch_start = exp_params["epoch_start"]
            if "num_epochs" in exp_params:
                total_epochs = exp_params["num_epochs"]

        if self.ctx.is_dryrun():
            logger.info("Detected dry run mode. Performing a single episode step only.")

        """ Perform the processing """
        for current_epoch in range(epoch_start, total_epochs + 1):
            callbacks.on_epoch_start(phase="process", epoch=current_epoch)
            provider = self.ctx["providers"][self.split_name]
            for current_step, (batch_inputs, batch_labels) in enumerate(provider):
                current_step = current_step + 1
                callbacks.on_step(inputs=batch_inputs, outputs=None, labels=batch_labels, mask=None, loss=None,
                                  step=current_step)
                if self.ctx.is_dryrun():
                    break
            callbacks.on_epoch_end(epoch=current_epoch)
            if self.ctx.is_dryrun():
                break
        logger.info("Finished processing for the experiment '%s' ", self.ctx["config"]["name"])
