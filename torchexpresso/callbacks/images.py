from torchexpresso.utils import mkdir_if_not_exists, store_json_to

from torchexpresso.callbacks import Callback
from torchvision.transforms import transforms
import os


class SaveImage(Callback):
    """
        Stores the images from the inputs.

        Inputs should be seperated in two listings of 'image' and 'file_name'.
    """

    def __init__(self, target_directory, split_name, name="img_save"):
        super().__init__(name)
        self.target_directory = target_directory
        self.split_name = split_name
        self.target_sub_directory = os.path.join(target_directory, split_name)
        if not os.path.exists(self.target_sub_directory):
            print("Create target sub-directory at: %s" % self.target_sub_directory)
            os.mkdir(self.target_sub_directory)
        self.transform = transforms.ToPILImage()

    def on_step(self, inputs, outputs, labels, mask, loss, step):
        for image_state, file_name in zip(inputs["image"], inputs["file_name"]):
            with self.transform(image_state) as image_pil:
                image_file = os.path.join(self.target_sub_directory, file_name)
                image_pil.save(image_file)


class SaveImageByLabel(Callback):
    """
        Stores the images structured by the labels. Labels should be a listing of dict with 'class_id'.

        Images are stored to a sub-directory called images with the directory name is the label.
        Annotations are not necessary, because implicitly given by the directory names.
    """

    def __init__(self, task, target_directory, split_name, name="img_save_by_label"):
        super().__init__(name)
        self.task = task
        self.target_directory = target_directory
        self.split_name = split_name
        self.num_images = 0
        self.annotations_directory = os.path.join(target_directory, "annotations")
        self.images_directory = os.path.join(target_directory, "images")
        self.transform = transforms.ToPILImage()
        self.images_split_dir = os.path.join(self.images_directory, self.split_name)
        self.current_epoch = 0
        self.annotations = []

        mkdir_if_not_exists(self.images_directory)
        mkdir_if_not_exists(self.annotations_directory)
        mkdir_if_not_exists(self.images_split_dir)

    def on_epoch_start(self, phase, epoch):
        print("Store epoch: " + str(epoch))
        self.current_epoch = epoch
        # Well, we "know" that epoch is the label
        mkdir_if_not_exists(os.path.join(self.images_split_dir, str(epoch)))

    def on_step(self, inputs, outputs, labels, mask, loss, step):
        for image, label in zip(inputs, labels):
            self.num_images = self.num_images + 1
            label_name = str(label["class_id"])
            file_name = "%s.png" % self.num_images
            image_directory = os.path.join(self.images_directory, self.split_name, label_name)
            self.annotations.append({"class_id": label["class_id"], "file_name": file_name})
            with self.transform(image) as image_pil:
                image_file = os.path.join(image_directory, file_name)
                image_pil.save(image_file)

    def on_epoch_end(self, epoch):
        store_json_to(self.annotations, self.annotations_directory, self.split_name + ".json")
