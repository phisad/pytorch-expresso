import torch
import logging
from torchexpresso import contexts
from torchexpresso.callbacks import CallbackRegistry
from torchexpresso.steps import TrainingStep, Step
from torchexpresso.savers import Saver, NoopSaver

logger = logging.getLogger(__file__)


class Trainer(object):

    @classmethod
    def from_config(cls, experiment_config: dict, train_split: str, dev_split: str):
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
            callbacks = CallbackRegistry()
        if saver is None:
            saver = NoopSaver()
        if step is None:
            step = self.ctx["step_fn"]
        epoch_start = self.ctx["epoch_start"]
        if len(callbacks) == 0:
            logger.info("No callbacks or saver registered!")

        """ Perform the training and validation """
        total_epochs = self.ctx["config"]["params"]["num_epochs"]
        for epoch in range(epoch_start, total_epochs + 1):
            self.__train(epoch, callbacks, step)
            self.__validate(epoch, callbacks, step)
            saver.save_checkpoint_if_best(self.ctx["model"], self.ctx["optimizer"], epoch, callbacks)
        logger.info("Finished training for the experiment '%s' ", self.ctx["config"]["name"])

    def __validate(self, current_epoch, callbacks, step: TrainingStep):
        self.ctx["model"].eval()
        with self.ctx["comet"].validate(), torch.no_grad():
            self.__perform_steps("validate", current_epoch, callbacks, step)

    def __train(self, current_epoch, callbacks: CallbackRegistry, step: TrainingStep):
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
            print("%s epoch %s: Step %s" % (current_phase, current_epoch, current_step), end="\r")
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
        print()
        callbacks.on_epoch_end(epoch=current_epoch)


class Predictor(object):

    @classmethod
    def from_config(cls, experiment_config, split_name):
        """
            @param experiment_config: a dictionary with all meta-information to perform the training
            @param split_name: Name of the dataset split on which to perform the prediction.
        """
        context = contexts.PredictionContext.from_config(experiment_config, [split_name])
        return cls(context, split_name)

    def __init__(self, context, split_name):
        self.ctx = context
        self.split_name = split_name

    def perform(self, callbacks: CallbackRegistry = None, step: Step = None):
        logger.info("Perform prediction for the experiment '%s' on '%s'", self.ctx["config"]["name"], self.split_name)
        if callbacks is None:
            callbacks = CallbackRegistry()
        if step is None:
            step = Step()
        """ Perform the test """
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
            callbacks.on_epoch_end(epoch=None)
