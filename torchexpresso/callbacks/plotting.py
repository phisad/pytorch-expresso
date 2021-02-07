import torch
from torchexpresso.utils import image_to_numpy

from torchexpresso.callbacks import Callback


class PlotOutputImage(Callback):
    """
        Plot N outputs (at index) of the last step in an episode.
    """

    def __init__(self, experiment, name="plt_img_output", label="output", num_images=3, index=None):
        super().__init__(name)
        self.experiment = experiment
        self.label = label
        self.index = index
        self.num_images = num_images
        self.last_outputs = []

    def on_epoch_start(self, phase, epoch):
        self.last_outputs = []

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.index:
            self.last_outputs = outputs[self.index]
        self.last_outputs = outputs

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        outputs = self.last_outputs[:self.num_images]
        outputs = [image_to_numpy(o) for o in outputs]
        for idx, output in enumerate(outputs):
            self.experiment.log_image(image_data=output, name="%s_%s" % (self.label, idx), step=epoch)


class PlotInputImage(Callback):
    """
        Plot N inputs (at index) of the last step in an episode.
    """

    def __init__(self, experiment, image_shape, name="plt_img_inputs", num_images=3, index=None):
        super().__init__(name)
        self.experiment = experiment
        self.image_shape = image_shape
        self.index = index
        self.num_images = num_images
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
        batch_size = inputs.size()[0]
        inputs = torch.reshape(inputs, [batch_size] + self.image_shape)
        self.last_inputs = inputs
        self.last_labels = labels

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        input_images = self.last_inputs[:self.num_images]
        input_images = [image_to_numpy(o) for o in input_images]
        input_labels = self.last_labels[:self.num_images]
        input_labels = [lbl.item() for lbl in input_labels]
        for idx, (image, label) in enumerate(zip(input_images, input_labels)):
            self.experiment.log_image(image_data=image, name="label_%s_%s" % (label, idx), step=epoch)
