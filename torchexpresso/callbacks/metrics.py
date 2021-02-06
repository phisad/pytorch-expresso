import torch

from torchexpresso.callbacks import Callback


class Metric(Callback):
    """
    Base class for callbacks that collect metrics
    """

    @torch.no_grad()
    def to_value(self):
        pass


class MetricsMetric(Metric):
    """
        Marker class for a metric operating on other metrics, which should have been loaded and calculated before.
    """


class AverageMetricsMetric(MetricsMetric):
    """
    Register multiple metrics and average their values on_epoch_end
    """

    def __init__(self, experiment, name, metrics: list, on_phase=None):
        super().__init__(name)
        self.experiment = experiment
        self.metrics = metrics
        self.on_phase = on_phase
        self.current_phase = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase

    @torch.no_grad()
    def to_value(self):
        total = len(self.metrics)
        return torch.true_divide(sum([metric.to_value() for metric in self.metrics]), total).item()

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        self.experiment.log_metric(self.name, self.to_value(), step=epoch)


""" 
Common Metrics 
"""


class AverageLossMetric(Metric):

    def __init__(self, experiment, name="epoch_loss", on_phase=None, index=None, context=None):
        super().__init__(name)
        self.experiment = experiment
        self.index = index
        self.context = context
        self.value = 0
        self.total = 0
        self.on_phase = on_phase
        self.current_phase = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        self.value = 0
        self.total = 0

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        if self.index is None:
            self.value += loss.item()
        else:
            self.value += loss[self.index].item()
        self.total = step

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.value, self.total)

    def on_epoch_end(self, epoch):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        if self.context:
            with self.experiment.context_manager(self.current_phase + "_" + self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)


class AverageClassActivation(Metric):
    """
        For classification (single correct label) problems we can measure the mean activation of the correct class.

        We would expect that this approaches one using a softmax classifier, because then the loss is zero.
    """

    def __init__(self, experiment, name="epoch_class_activation", on_phase=None, index=None, context=None):
        super().__init__(name)
        self.experiment = experiment
        self.index = index
        self.context = context
        self.value = 0
        self.total = 0
        self.on_phase = on_phase
        self.current_phase = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        self.value = 0
        self.total = 0

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        if self.index:
            outputs = outputs[self.index]
            labels = labels[self.index]
        predictions = torch.softmax(outputs, 1)
        cls_values = torch.as_tensor([pred[idx] for pred, idx in zip(predictions, labels)])
        self.value += torch.sum(cls_values)
        self.total += len(cls_values)

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.value, self.total)

    def on_epoch_end(self, epoch):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        if self.context:
            with self.experiment.context_manager(self.current_phase + "_" + self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)


class BinaryAccuracyMetric(Metric):
    """
        For binary output units this applies sigmoid on the logits and rounds towards zero or one.
    """

    def __init__(self, experiment, name="epoch_binary_accuracy", on_phase=None, index=None, context=None):
        super().__init__(name)
        self.experiment = experiment
        self.index = index
        self.context = context
        self.value = 0
        self.total = 0
        self.on_phase = on_phase
        self.current_phase = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        self.value = 0
        self.total = 0

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        if self.index is not None:
            # B x L
            outputs = outputs[:, self.index]
            labels = labels[:, self.index]
        predictions = torch.sigmoid(outputs)
        rounded = torch.round(predictions).long()
        labels = labels.long()
        self.total += labels.size(0)  # single value tensor item moves to cpu
        self.value += (rounded == labels).sum().item()  # single value tensor item moves to cpu

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.value, self.total).item()

    def on_epoch_end(self, epoch):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        if self.context:
            with self.experiment.context_manager(self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)


class CategoricalAccuracyMetric(Metric):

    def __init__(self, experiment, name="epoch_accuracy", on_phase=None, index=None, context=None):
        super().__init__(name)
        self.experiment = experiment
        self.index = index
        self.context = context
        self.value = 0
        self.total = 0
        self.on_phase = on_phase
        self.current_phase = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        self.value = 0
        self.total = 0

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        if self.index is not None:
            outputs = outputs[self.index]
            labels = labels[self.index]
        # B x V x L
        _, predicted = torch.max(outputs, 1)
        self.total += labels.size(0)  # single value tensor item moves to cpu
        self.value += (predicted == labels).sum().item()  # single value tensor item moves to cpu

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.value, self.total).item()

    def on_epoch_end(self, epoch):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        if self.context:
            with self.experiment.context_manager(self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)
