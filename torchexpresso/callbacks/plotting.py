import torch
from torchexpresso.utils import image_to_numpy

from torchexpresso.callbacks import Callback


class PlotOutputImage(Callback):
    """
        Plot N outputs (at index) of the last step in an episode.
    """

    def __init__(self, experiment, name="plt_img_output", num_images=3, index=None, ignore_lbl=False):
        super().__init__(name)
        self.experiment = experiment
        self.index = index
        self.num_images = num_images
        self.ignore_lbl = ignore_lbl
        self.last_outputs = []
        self.last_labels = []

    def on_epoch_start(self, phase, epoch):
        self.last_outputs = []
        self.last_labels = []

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.index:
            outputs = outputs[self.index]
            labels = labels[self.index]
        self.last_outputs = outputs
        self.last_labels = labels

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        outputs = self.last_outputs[:self.num_images]
        output_images = [image_to_numpy(o) for o in outputs]
        if self.ignore_lbl:  # Note: Might be necessary for image without label e.g. when reconstructing
            for idx, image in enumerate(output_images):
                self.experiment.log_image(image_data=image, name="%s_out" % idx, step=epoch)
        else:
            labels = self.last_labels[:self.num_images]
            labels = [lbl.item() for lbl in labels]
            for idx, (image, label) in enumerate(zip(output_images, labels)):
                self.experiment.log_image(image_data=image, name="%s_%s_out" % (idx, label), step=epoch)


class PlotInputImage(Callback):
    """
        Plot N inputs (at index) of the last step in an episode.
    """

    def __init__(self, experiment, image_shape=None, name="plt_img_inputs", num_images=3, index=None, ignore_lbl=False):
        super().__init__(name)
        self.experiment = experiment
        self.image_shape = image_shape
        self.index = index
        self.num_images = num_images
        self.ignore_lbl = ignore_lbl
        self.last_inputs = []
        self.last_labels = []

    def on_epoch_start(self, phase, epoch):
        self.last_inputs = []
        self.last_labels = []

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.index:
            inputs = inputs[self.index]
            labels = labels[self.index]
        # Inputs are after model.prepare(), so we reshape if necessary
        if self.image_shape:
            batch_size = inputs.size()[0]
            inputs = torch.reshape(inputs, [batch_size] + self.image_shape)
        self.last_inputs = inputs
        self.last_labels = labels

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        input_images = self.last_inputs[:self.num_images]
        input_images = [image_to_numpy(o) for o in input_images]
        if self.ignore_lbl:  # Note: Might be necessary for image without label e.g. when reconstructing
            for idx, image in enumerate(input_images):
                self.experiment.log_image(image_data=image, name="%s_in" % idx, step=epoch)
        else:
            input_labels = self.last_labels[:self.num_images]
            input_labels = [lbl.item() for lbl in input_labels]
            for idx, (image, label) in enumerate(zip(input_images, input_labels)):
                self.experiment.log_image(image_data=image, name="%s_%s_in" % (idx, label), step=epoch)
