import torch
import logging

logger = logging.getLogger(__file__)


def has_method(o, name):
    return callable(getattr(o, name, None))


class Step(object):

    def prepare(self, model, batch_inputs, device, step: int = None):  # noqa
        if has_method(model, "prepare"):
            return model.prepare(batch_inputs, device)
        return batch_inputs

    def forward(self, model, batch_inputs, device, step: int = None):  # noqa
        batch_outputs = model(batch_inputs, device)
        return batch_outputs, batch_inputs


class TrainingStep(Step):

    def __init__(self):
        self.current_phase = None
        self.current_epoch = None

    def on_epoch_start(self, phase: str, epoch: int):
        self.current_phase = phase
        self.current_epoch = epoch

    def loss(self, loss_fn, batch_outputs, batch_labels, device, step: int = None):  # noqa
        batch_labels = torch.stack(batch_labels).to(device)
        loss = loss_fn(batch_outputs, batch_labels)
        if self.current_phase == "train":
            loss.backward()
        return loss, batch_labels


class SequenceLossTrainingStep(TrainingStep):
    """
        Notice: If there are pad tokens with id=0, then this training step should be used with
        torch.nn.CrossEntropyLoss(ignore_index=0) to ignore these pad tokens on loss computation.
    """

    def forward(self, model, batch_inputs, device, step: int = None):  # noqa
        batch_outputs = model(batch_inputs, device)
        # Output is L x B x V, but need B x V x L for loss
        batch_outputs = batch_outputs.permute(dims=[1, 2, 0])
        return batch_outputs, batch_inputs

    def loss(self, loss_fn, batch_outputs, batch_labels, device, step: int = None):  # noqa
        batch_labels = torch.stack(batch_labels).to(device)
        loss = loss_fn(batch_outputs, batch_labels)
        if self.current_phase == "train":
            loss.backward()
        return loss, batch_labels


class MultiTaskTrainingStep(TrainingStep):
    """
        A multi task training step allows to train a single model for multiple tasks.
        For example given an image the same person classifier might predict age and gender.

        Each of the tasks have non-overlapping labels (in constrast to MultiLabelTrainingStep).
        For example a person is at a certain age or not.

        This is technically accumulating the individual losses and performing a single back-prop step.
    """

    def loss(self, loss_fn, multi_batch_outputs: list, multi_batch_labels: list, device, step: int = None):  # noqa
        """
        :param multi_batch_outputs: a list of batch tensors;
            with each batch tensor in the same order as the labels listing.
            For example a three label multi-task with batch size 32 has a list of [tensor(32),tensor(32),tensor(32)].
        :param multi_batch_labels: a batch list of label tensor;
            with each label tensor in the same order as the outputs tensors.
            For example a three label multi-task with batch size 32 has a 32-element list of [tensor(3),...,tensor(3)]
        """
        num_labels = len(multi_batch_labels[0])
        if num_labels == 1:
            logger.warning("Using MultiLossTrainingStep, but batch_labels are single labels. "
                           "Please use TrainingStep for single label tasks.")
        num_outputs = len(multi_batch_outputs)
        if num_labels != num_outputs:
            raise Exception("Label count '%s' does not model outputs '%s'. "
                            "Please provide a label for each output." % (num_labels, num_outputs))

        multi_batch_labels = torch.stack(multi_batch_labels)
        multi_batch_labels = multi_batch_labels.permute(1, 0)  # TODO handle multi-dim labels
        multi_loss = []
        for idx, batch_outputs in enumerate(multi_batch_outputs):
            batch_labels = multi_batch_labels[idx]
            multi_loss.append(loss_fn(batch_outputs, batch_labels))
        if self.current_phase == "train":
            accumulated_loss = torch.stack(multi_loss)
            accumulated_loss = torch.sum(accumulated_loss)
            accumulated_loss.backward()
        return multi_loss, multi_batch_labels


class MultiLabelTrainingStep(TrainingStep):
    """
        A multi label training step allows to train a model which predict multiple labels for the same task.
        For example in the detection task there might be multiple classes on the same image.

        This is technically performing N (number of labels) individual binary classifications.

        Simply configure the loss_fn for this with the default TrainingStep:

            "loss_fn": {
              "package": "torch.nn",
              "class": "BCEWithLogitsLoss",
              "kwargs": {
                "reduction": "sum"
              }
            }
    """


class MultiLossTrainingStep(TrainingStep):
    """
        A multi loss training stepp allows to train multiple models on seperate tasks.
        This is useful to train multiple networks for difference tasks in a single run.

        This is technically computing each loss individually and performing for each loss an individual back-prop.

        Note: This is not effecient to train a single model on multiple tasks (see MultiTaskTrainingStep).
    """

    def __init__(self, retain_graph=False):
        """
        :param retain_graph: is slower, but necessary, when the losses are applied against the same graph
            (shared model parameters) Otherwise the graph will be removed after the first loss computation.
            The 'retain_graph' option is not necessary, when the graph do not share parameters (seperate models).
        """
        super().__init__()
        self.retain_graph = retain_graph

    def loss(self, loss_fn, multi_batch_outputs: list, multi_batch_labels: list, device, step: int = None):  # noqa
        """
        :param multi_batch_outputs: a list of batch tensors;
            with each batch tensor in the same order as the labels listing.
            For example a three label multi-task with batch size 32 has a list of [tensor(32),tensor(32),tensor(32)].
        :param multi_batch_labels: a batch list of label tensor;
            with each label tensor in the same order as the outputs tensors.
            For example a three label multi-task with batch size 32 has a 32-element list of [tensor(3),...,tensor(3)]
        """
        num_labels = len(multi_batch_labels[0])
        if num_labels == 1:
            logger.warning("Using MultiLossTrainingStep, but batch_labels are single labels. "
                           "Please use TrainingStep for single label tasks.")
        num_outputs = len(multi_batch_outputs)
        if num_labels != num_outputs:
            raise Exception("Label count '%s' does not model outputs '%s'. "
                            "Please provide a label for each output." % (num_labels, num_outputs))

        multi_batch_labels = torch.stack(multi_batch_labels)
        multi_batch_labels = multi_batch_labels.permute(1, 0)  # TODO handle multi-dim labels
        multi_loss = []
        for idx, batch_outputs in enumerate(multi_batch_outputs):
            batch_labels = multi_batch_labels[idx]
            multi_loss.append(loss_fn(batch_outputs, batch_labels))
        if self.current_phase == "train":
            for idx, loss in enumerate(multi_loss):
                if self.retain_graph:
                    loss.backward(retain_graph=True if idx < num_labels else False)
                else:
                    loss.backward()
        # The optimizer is applying all the gradient values stored in the tensors.
        # So we can indeed share a single optimizer here, which is simply applying the values.
        # Nevertheless, depending on the optimizer implementation there might be side-effects.
        return multi_loss, multi_batch_labels
