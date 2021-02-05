import torch
import logging
import os

logger = logging.getLogger(__file__)


class Saver(object):

    def save_checkpoint_if_best(self, model, optimizer, epoch, metrics):
        raise NotImplementedError()

    def save_checkpoint(self, model, optimizer, epoch):
        raise NotImplementedError()


class NoopSaver(Saver):

    def save_checkpoint_if_best(self, model, optimizer, epoch, metrics):
        pass

    def save_checkpoint(self, model, optimizer, epoch):
        pass


class CheckpointSaver(Saver):

    def __init__(self, checkpoint_top_dir, experiment_name, model_name, metric, mode="highest"):
        self.checkpoint_dir = os.path.join(checkpoint_top_dir, experiment_name)
        self.model_name = model_name
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

    def save_checkpoint_if_best(self, model, optimizer, epoch, metrics):
        epoch_value = metrics[self.metric].to_value()
        if self.comparator(epoch_value, self.best_value):
            logger.info("Save checkpoint at epoch %s: epoch_value %s best_value (%.3f %s %.3f) [%s]" %
                        (str(epoch), self.comparator_string, epoch_value, self.comparator_string, self.best_value,
                         self.checkpoint_dir))
            self.best_value = epoch_value
            self.save_checkpoint(model, optimizer, epoch)

    def save_checkpoint(self, model, optimizer, epoch):
        if not os.path.exists(self.checkpoint_dir):
            logger.info("Created experiment checkpoint directory at %s", self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir)
        torch.save({
            'epoch': epoch,
            'arch': self.model_name,
            'state_dict': model.state_dict(),
            'best_value': self.best_value,
            'best_value_metric': self.metric,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(self.checkpoint_dir, "model_best.pth.tar"))

    @staticmethod
    def load_checkpoint(model, checkpoint_dir, experiment_name):
        experiment_checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        experiment_checkpoint_path = os.path.join(experiment_checkpoint_dir, "model_best.pth.tar")
        if not os.path.exists(experiment_checkpoint_path):
            raise Exception("Cannot find experiment checkpoint at %s", experiment_checkpoint_path)
        checkpoint = torch.load(experiment_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return checkpoint
