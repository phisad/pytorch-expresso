from PIL import Image
import torch
import os
import json
import numpy as np


def determine_sub_directory(directory, sub_directory_name, to_read=True):
    """
        @param directory: to look for the source file
        @param sub_directory_name: the filename to look for when a directory is given
        @param to_read: check for file existence, when true
    """
    sub_directory = os.path.join(directory, sub_directory_name)
    if os.path.isdir(directory):
        if sub_directory_name is None:
            raise Exception("Cannot determine sub-directory without sub_directory_name")
    if to_read and not os.path.isdir(sub_directory):
        raise Exception("There is no such sub-directory to read: " + sub_directory)
    return sub_directory


def mkdir_if_not_exists(directory):
    if not os.path.exists(directory):
        print("Create directory at: %s" % directory)
        os.mkdir(directory)


def exists_sub_directory(directory, sub_directory_name):
    sub_directory = os.path.join(directory, sub_directory_name)
    return exists_directory(sub_directory)


def exists_directory(directoy):
    if os.path.exists(directoy):
        return True
    return False


def determine_file_path(directory_or_file, lookup_filename=None, to_read=True):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
        @param to_read: check for file existence, when true
    """
    file_path = directory_or_file
    if os.path.isdir(directory_or_file):
        if lookup_filename is None:
            raise Exception("Cannot determine source file in directory without lookup_filename")
        file_path = os.path.join(directory_or_file, lookup_filename)
    if to_read and not os.path.isfile(file_path):
        raise Exception("There is no such file in the directory to read: " + file_path)
    return file_path


def load_json_from(directory_or_file, lookup_filename=None):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(directory_or_file, lookup_filename)
    # print("Loading JSON from " + file_path)
    with open(file_path) as json_file:
        json_content = json.load(json_file)
    return json_content


def store_json_to(json_content, directory_or_file, lookup_filename=None,
                  json_indent=4, json_sort_keys=True, verbose=False):
    """
        @param json_content: the json data to store
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(directory_or_file, lookup_filename, to_read=False)
    if verbose:
        print("Persisting JSON to " + file_path)
    with open(file_path, "w") as json_file:
        json.dump(json_content, json_file, indent=json_indent, sort_keys=json_sort_keys)
    return file_path


def load_aggregate_model(model, checkpoint_path, fine_tune):
    """
    If a model builds upon another one, then we load the other models weights here.

    :param model: to load the pre-trained weights for
    :param checkpoint_path: to the checkpoint of the pre-trained other model
    :param fine_tune: if to enable training for the pre-trained model weights
    """
    if not os.path.exists(checkpoint_path):
        raise Exception("Cannot find model checkpoint at %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False


def pad_inputs(inputs: list, max_length: int, pad_value, dtype, device):
    """
    Pad the inputs to a fixed size. The entries in the batch must be shorter than max_length.

    :param inputs: to pad to max_length (not just to the longest in the batch)
    :param max_length: the length upto which padding is applied
    :param pad_value: the value to use for padding
    :param dtype: of the padded inputs
    :param device: of the padded inputs

    :return: the padded inputs
    """
    max_size = inputs[0].size()
    trailing_dims = max_size[1:]
    out_dims = (len(inputs), max_length) + trailing_dims
    inputs_padded = torch.full(size=out_dims, fill_value=pad_value, dtype=dtype, device=device)
    # We need to iterate here, because each tensor has individual length
    for idx, tensor in enumerate(inputs):
        length = len(tensor)
        inputs_padded[idx, :length, ...] = tensor
    return inputs_padded


def load_image_file_names(images_directory, file_ending=".png"):
    return [file_name for file_name in os.listdir(images_directory) if file_name.endswith(file_ending)]


def load_image_from_dir(image_directory, image_file_name, image_transforms=None):
    image_path = os.path.join(image_directory, image_file_name)
    with Image.open(image_path) as image:
        if image_transforms:
            return image_transforms(image)
        return image


def image_to_numpy(image: torch.Tensor, normalize: tuple = None, height_first=True):
    """
    Convert a pytorch (image) tensor to numpy e.g. for plotting. The tensor will detached and copied to cpu.

    :param image: a pytorch tensor (C,H,W)
    :param normalize: a tuple (std,mean) to denormalize
    :param height_first: if true then returns (H,W,C) else (W,H,C)
    :return: a numpy array with values in [0,255]
    """
    image = image.detach().cpu().numpy()
    if normalize:
        image = image * normalize[0] + normalize[1]
    image = (image * 255).astype(np.uint8)
    # Note: The pyglet viewer return (H, W, C), but the actual engine is operating on (W, H, C)
    if height_first:
        image = image.transpose((1, 2, 0))
    else:
        image = image.transpose((2, 1, 0))
    return image


def image_to_patches_2d(image: torch.Tensor, num_patches: int):
    """
    Returns N*N patches from the input image.

    Note: Only tested when num_patches is a whole divisor of H or W
    e.g. for an image of 100 x 100 pixels returns 5 * 5 patches of size 20 x 20, when num_patches = 5.

    :param image: with dimensions (C, H, W)
    :param num_patches: number of patches N (possibly a multiple of H and W)
    :return: the patches in dimension (N*N) x C x H x W
    """
    image_channels = image.size()[0]
    image_height = image.size()[1]
    patch_size = image_height // num_patches
    patches = image
    for channel in range(1, image_channels + 1):
        patches = patches.unfold(channel, patch_size, patch_size)
    # Unfold returns a "changed" view/indexing on the "unchanged" data,
    # thus we call contiguous() to re-sync the indexing with the view
    # C x N x N x 1 x W x H -> C x (N*N) x W x H
    patches = patches.contiguous().view(image_channels, -1, patch_size, patch_size)
    # Permute shifts the actual data, returning a contiguous() view
    # We use (N*N) as batch dimension; reset height as first dim
    # C x (N*N) x W x H -> (N*N) x C x H x W
    patches = patches.permute(dims=(1, 0, 3, 2))
    return patches
