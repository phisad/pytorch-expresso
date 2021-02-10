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

    def is_applicable(self):
        return self.current_phase in self.on_phase

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if not self.is_applicable():
            return
        self.value = 0
        self.total = 0
        self._guarded_on_epoch_start(phase, epoch)

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if not self.is_applicable():
            return
        self._guarded_on_step(inputs, outputs, labels, mask, loss, step)

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if not self.is_applicable():
            return
        self._guarded_on_epoch_end(epoch)

    def _guarded_on_epoch_start(self, phase, epoch):
        """ Phase-guarded invocation"""
        pass

    def _guarded_on_step(self, inputs, outputs, labels, mask, loss, step):
        """ Phase-guarded invocation"""
        pass

    def _guarded_on_epoch_end(self, epoch):
        """ Phase-guarded invocation"""
        pass

    @torch.no_grad()
    def to_value(self):
        pass


class IndexMetric(Metric):

    def __init__(self, name, on_phase: list, index=None):
        super().__init__(name, on_phase)
        # Accept lists or ints
        self.index = index
        # Convert ints to lists
        if isinstance(index, int):
            self.index = [index]
        self.sensitivity = []

    def _guarded_on_step(self, inputs, outputs, labels, mask, loss, step):
        """ Phase-guarded invocation"""
        if self.index:
            if "outputs" in self.sensitivity:
                outputs = self._select_at_index(outputs)
            if "labels" in self.sensitivity:
                labels = self._select_at_index(labels)
            if "losses" in self.sensitivity:
                loss = self._select_at_index(loss)
        self._index_guarded_on_step(inputs, outputs, labels, mask, loss, step)

    def _index_guarded_on_step(self, inputs, outputs, labels, mask, loss, step):
        """ Phase-guarded invocation with index injection"""
        pass

    def _select_at_index(self, entry):
        # this should work for lists-in-lists and tuples and even for tensors, when batch is last dimension
        for idx in self.index:
            try:
                entry = entry[idx]
            except Exception as e:
                raise Exception("%s[%s]: %s [idx: %s]" % (self.__class__, self.name, e, idx))
        return entry

    def set_index_sensitivity(self, sensitivity: list):
        self.sensitivity = sensitivity


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
    def _guarded_on_epoch_end(self, epoch):
        self.experiment.log_metric(self.name, self.to_value(), step=epoch)


""" 
Common Metrics 
"""


class AverageLossMetric(IndexMetric):

    def __init__(self, experiment, name="epoch_loss",
                 on_phase: list = None, index=None, context=None):
        super().__init__(name, on_phase, index)
        self.experiment = experiment
        self.context = context
        self.set_index_sensitivity(["losses"])

    @torch.no_grad()
    def _index_guarded_on_step(self, inputs, outputs, labels, mask, loss, step):
        self.value += loss.item()
        self.total = step

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.value, self.total)

    @torch.no_grad()
    def _guarded_on_epoch_end(self, epoch):
        if self.context:
            with self.experiment.context_manager(self.current_phase + "_" + self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)


class AverageClassActivation(IndexMetric):
    """
        For classification (single correct label) problems we can measure the mean activation of the correct class.

        We would expect that this approaches one using a softmax classifier, because then the loss is zero.
    """

    def __init__(self, experiment, name="epoch_class_activation",
                 on_phase: list = None, index=None, context=None):
        super().__init__(name, on_phase, index)
        self.experiment = experiment
        self.context = context
        self.set_index_sensitivity(["outputs", "labels"])

    @torch.no_grad()
    def _index_guarded_on_step(self, inputs, outputs, labels, mask, loss, step):
        predictions = torch.softmax(outputs, 1)
        cls_values = torch.as_tensor([pred[idx] for pred, idx in zip(predictions, labels)])
        self.value += torch.sum(cls_values)
        self.total += len(cls_values)

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.value, self.total)

    @torch.no_grad()
    def _guarded_on_epoch_end(self, epoch):
        if self.context:
            with self.experiment.context_manager(self.current_phase + "_" + self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)


class BinaryAccuracyMetric(IndexMetric):
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
    def _guarded_on_step(self, inputs, outputs, labels, mask, loss, step):
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

    def _guarded_on_epoch_end(self, epoch):
        if self.context:
            with self.experiment.context_manager(self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)


class CategoricalAccuracyMetric(IndexMetric):

    def __init__(self, experiment, name="epoch_accuracy",
                 on_phase: list = None, index=None, context=None):
        super().__init__(name, on_phase, index)
        self.experiment = experiment
        self.context = context
        self.set_index_sensitivity(["outputs", "labels"])

    @torch.no_grad()
    def _index_guarded_on_step(self, inputs, outputs, labels, mask, loss, step):
        # B x V x L
        _, predicted = torch.max(outputs, 1)
        self.total += labels.size(0)  # single value tensor item moves to cpu
        self.value += (predicted == labels).sum().item()  # single value tensor item moves to cpu

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.value, self.total).item()

    def _guarded_on_epoch_end(self, epoch):
        if self.context:
            with self.experiment.context_manager(self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)


class CategoricalAccuracyMatrix(IndexMetric):

    def __init__(self, experiment, class_names, name="epoch_accuracy_matrix", on_phase=None, index=None):
        super().__init__(name, on_phase, index)
        self.experiment = experiment
        self.num_classes = len(class_names)
        self.confusion_matrix = comet_ml.ConfusionMatrix(labels=class_names)
        self.labels = None
        self.predictions = None
        self.set_index_sensitivity(["outputs"])

    @torch.no_grad()
    def _guarded_on_epoch_start(self, phase, epoch):
        self.labels = None
        self.predictions = None

    @torch.no_grad()
    def _index_guarded_on_step(self, inputs, outputs, labels, mask, loss, step):
        """
        @param labels: the true labels for the samples.
        @param outputs: the raw output of the model. A winner function (np.argmax) is applied on compute_matrix()
        """
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
    def _guarded_on_epoch_end(self, epoch):
        self.confusion_matrix.compute_matrix(self.labels.cpu().numpy(), self.predictions.numpy())
        if epoch:
            matrix_title = "%s, Epoch #%s" % (self.name, epoch)
            matrix_file = "%s-%s-%03d.json" % (self.current_phase, self.name, epoch)
        else:
            matrix_title = self.name
            matrix_file = "%s-%s.json" % (self.current_phase, self.name)
        self.experiment.log_confusion_matrix(matrix=self.confusion_matrix, title=matrix_title, file_name=matrix_file)
