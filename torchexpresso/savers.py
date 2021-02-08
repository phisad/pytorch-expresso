from collections import OrderedDict

import torch
import logging
import os

logger = logging.getLogger(__file__)


def load_checkpoint(checkpoint_dir: str, file_name: str):
    checkpoint_path = os.path.join(checkpoint_dir, "%s.pth.tar" % file_name)
    return load_checkpoint_from_path(checkpoint_path)


def load_checkpoint_from_path(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise Exception("Cannot find checkpoint at %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


def save_checkpoint(checkpoint_dir: str, file_name: str, checkpoint: dict):
    if not os.path.exists(checkpoint_dir):
        logger.info("Created experiment checkpoint directory at %s", checkpoint_dir)
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, os.path.join(checkpoint_dir, "%s.pth.tar" % file_name))


class Saver(object):

    def __init__(self, name):
        self.name = name

    def on_epoch_end(self, model, optimizer, epoch, metrics):
        raise NotImplementedError()


class SaverRegistry(Saver):
    """
        Register one or more callbacks for callback method invocation. Keeps the order of added callbacks.
    """

    def __init__(self, name="saver_registry"):
        super().__init__(name)
        self.savers = OrderedDict()

    @torch.no_grad()
    def on_epoch_end(self, model, optimizer, epoch, metrics):
        for srv in self.savers.values():
            srv.on_epoch_end(model, optimizer, epoch, metrics)

    def __contains__(self, o):
        return self.savers.__contains__(o)

    def __getitem__(self, key):
        return self.savers[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Saver):
            raise Exception("Value to add is no Saver, but %s" % value.__class__)
        self.savers[key] = value

    def __iter__(self):
        return self.savers.__iter__()

    def __len__(self):
        return len(self.savers)


class NoopSaver(Saver):

    def on_epoch_end(self, model, optimizer, epoch, metrics):
        pass


class ModelSaver(Saver):

    def __init__(self, model_config, task_config, checkpoint_dir, file_name="model", name="epoch_saver"):
        super().__init__(name)
        self.task_config = task_config
        self.model_config = model_config
        self.checkpoint_dir = checkpoint_dir
        self.file_name = file_name

    def on_epoch_end(self, model, optimizer, epoch, metrics):
        save_checkpoint(self.checkpoint_dir, self.file_name, {
            'cp-epoch': epoch,
            'cp-model': self.model_config,
            'cp-task': self.task_config,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })


class BestModelSaver(Saver):

    def __init__(self, model_config, task_config, checkpoint_dir, metric,
                 mode="highest", file_name="model_best", name="best_saver"):
        super().__init__(name)
        self.task_config = task_config
        self.model_config = model_config
        self.checkpoint_dir = checkpoint_dir
        self.file_name = file_name
        self.metric = metric
        if mode == "highest":
            self.comparator = lambda x, y: x > y
            self.best_value = 0
            self.comparator_string = ">"
        if mode == "lowest":
            import math
            self.comparator = lambda x, y: x < y
            self.best_value = math.inf
            self.comparator_string = "<"

    def on_epoch_end(self, model, optimizer, epoch, metrics):
        epoch_value = metrics[self.metric].to_value()
        if self.comparator(epoch_value, self.best_value):
            logger.info("Save checkpoint at epoch %s: epoch_value %s best_value (%.3f %s %.3f) [%s]" %
                        (str(epoch), self.comparator_string, epoch_value, self.comparator_string, self.best_value,
                         self.checkpoint_dir))
            self.best_value = epoch_value
            save_checkpoint(self.checkpoint_dir, self.file_name, {
                'cp-epoch': epoch,
                'cp-model': self.model_config,
                'cp-task': self.task_config,
                'cp-value': self.best_value,
                'cp-metric': self.metric,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })
