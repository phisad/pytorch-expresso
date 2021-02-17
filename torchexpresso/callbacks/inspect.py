from torchexpresso.callbacks import CallbackRegistry

from torchexpresso.callbacks.metrics import Metric

import torch

import logging

logger = logging.getLogger(__file__)


class ModelGradientMetric(Metric):
    """
        Calculates an accumulated gradient score for all parameters in a model.
    """

    def __init__(self, experiment, model, name="grad_model", log_on_step=False):
        """
        @param model: the model with parameters (only weights are logged)
        """
        super().__init__(name, on_phase=["train"])
        self.log_on_step = log_on_step
        self.experiment = experiment
        self.model = model
        self.current_step = 0
        # Only weights which require grads (param[0] is name, param[1] is weight tensor)
        self.model_parameters = []
        for param in list(model.named_parameters()):
            if param[1].requires_grad:
                if "weight" in param[0]:
                    self.model_parameters.append(param)

    @torch.no_grad()
    def _guarded_on_epoch_start(self, phase, epoch):
        # We collect individual grads for the histogram on epoch end
        # self.parameter_grads = dict([("grad." + param[0], torch.zeros_like(param[1]))
        #                             for param in self.model_parameters])
        ...

    @torch.no_grad()
    def _guarded_on_step(self, inputs, outputs, labels, mask, loss, step):
        step_grads = [param[1].grad.abs().sum() for param in self.model_parameters]
        step_grads = torch.stack(step_grads).sum().cpu()
        step_grads = step_grads.item()
        self.value += step_grads
        self.total += 1
        self.current_step += 1  # We keep track of an ongoing step, otherwise repeats in each epoch
        if self.log_on_step:
            self.experiment.log_metric(name="step_" + self.name, value=step_grads, step=self.current_step)

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.value, self.total)

    @torch.no_grad()
    def _guarded_on_epoch_end(self, epoch):
        self.experiment.log_metric(name="epoch_" + self.name, value=self.to_value(), step=epoch)


class ParameterGradientMetricRegistry(Metric, CallbackRegistry):
    """
    Register multiple metrics and average their values on_epoch_end.

    The other metrics *are* loaded and calculated before.
    """

    def __init__(self, experiment, name, metrics: list, on_phase: list = None):
        super().__init__(name, on_phase)
        self.experiment = experiment
        self.metrics = metrics
        # TODO Create metrics here automatically

    @torch.no_grad()
    def to_value(self):
        total = len(self.metrics)
        return torch.true_divide(sum([metric.to_value() for metric in self.metrics]), total).item()

    @torch.no_grad()
    def _guarded_on_epoch_end(self, epoch):
        self.experiment.log_metric(self.name, self.to_value(), step=epoch)
