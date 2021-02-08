import comet_ml
import torch
import torch.nn.functional as functional

from torchexpresso.callbacks import Callback


class Metric(Callback):
    """
    Base class for callbacks that collect metrics
    """

    def __init__(self, name, on_phase: list):
        super().__init__(name)
        # As default, we apply metrics only during training
        if on_phase is None:
            on_phase = ["train", "validate"]
        self.on_phase = on_phase
        self.current_phase = None
        self.value = 0
        self.total = 0

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if not self.is_applicable():
            return
        self.value = 0
        self.total = 0

    def is_applicable(self):
        return self.current_phase in self.on_phase

    @torch.no_grad()
    def to_value(self):
        pass


class AverageMetricsMetric(Metric):
    """
    Register multiple metrics and average their values on_epoch_end.

    The other metrics should have been loaded and calculated before.
    """

    def __init__(self, experiment, name, metrics: list, on_phase: list = None):
        super().__init__(name, on_phase)
        self.experiment = experiment
        self.metrics = metrics

    @torch.no_grad()
    def to_value(self):
        total = len(self.metrics)
        return torch.true_divide(sum([metric.to_value() for metric in self.metrics]), total).item()

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if not self.is_applicable():
            return
        self.experiment.log_metric(self.name, self.to_value(), step=epoch)


""" 
Common Metrics 
"""


class AverageLossMetric(Metric):

    def __init__(self, experiment, name="epoch_loss",
                 on_phase: list = None, index=None, context=None):
        super().__init__(name, on_phase)
        self.experiment = experiment
        self.index = index
        self.context = context

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if not self.is_applicable():
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
        if not self.is_applicable():
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

    def __init__(self, experiment, name="epoch_class_activation",
                 on_phase: list = None, index=None, context=None):
        super().__init__(name, on_phase)
        self.experiment = experiment
        self.index = index
        self.context = context

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if not self.is_applicable():
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
        if not self.is_applicable():
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

    def __init__(self, experiment, name="epoch_binary_accuracy",
                 on_phase: list = None, index=None, context=None):
        super().__init__(name, on_phase)
        self.experiment = experiment
        self.index = index
        self.context = context

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if not self.is_applicable():
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
        if not self.is_applicable():
            return
        if self.context:
            with self.experiment.context_manager(self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)


class CategoricalAccuracyMetric(Metric):

    def __init__(self, experiment, name="epoch_accuracy",
                 on_phase: list = None, index=None, context=None):
        super().__init__(name, on_phase)
        self.experiment = experiment
        self.index = index
        self.context = context

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if not self.is_applicable():
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
        if not self.is_applicable():
            return
        if self.context:
            with self.experiment.context_manager(self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)


class CategoricalAccuracyMatrix(Metric):

    def __init__(self, experiment, class_names, name="epoch_accuracy_matrix", on_phase=None, index=None):
        super().__init__(name, on_phase)
        self.experiment = experiment
        self.num_classes = len(class_names)
        self.confusion_matrix = comet_ml.ConfusionMatrix(labels=class_names)
        self.index = index
        self.labels = None
        self.predictions = None

    @torch.no_grad()
    def on_epoch_start(self, phase, epoch):
        super().on_epoch_start(phase, epoch)
        if not self.is_applicable():
            return
        self.labels = None
        self.predictions = None

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        """
        @param labels: the true labels for the samples.
        @param outputs: the raw output of the model. A winner function (np.argmax) is applied on compute_matrix()
        """
        if not self.is_applicable():
            return
        if self.index is not None:
            outputs = outputs[self.index]
        if self.predictions is not None:
            self.predictions = torch.cat((self.predictions, outputs.detach().clone().cpu()))
        else:
            self.predictions = outputs.detach().clone().cpu()
        if self.labels is not None:
            self.labels = torch.cat(
                (self.labels, torch.stack([functional.one_hot(v, self.num_classes) for v in labels])))
        else:
            self.labels = torch.stack([functional.one_hot(v, self.num_classes) for v in labels])

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if not self.is_applicable():
            return
        self.confusion_matrix.compute_matrix(self.labels.cpu().numpy(), self.predictions.numpy())
        if epoch:
            matrix_title = "Confusion Matrix, Epoch #%s" % epoch
            matrix_file = "%s-confusion-matrix-%03d.json" % (self.current_phase, epoch)
        else:
            matrix_title = "Confusion Matrix"
            matrix_file = "%s-confusion-matrix.json" % self.current_phase
        self.experiment.log_confusion_matrix(matrix=self.confusion_matrix, title=matrix_title, file_name=matrix_file)
