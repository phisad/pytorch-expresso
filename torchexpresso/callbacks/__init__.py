from collections import OrderedDict

import torch


class Callback(object):
    """
    Base class for callbacks.
    """

    def __init__(self, name):
        self.name = name

    def on_epoch_start(self, phase, epoch):
        pass

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        pass

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        pass


class CallbackRegistry(Callback):
    """
        Register one or more callbacks for callback method invocation. Keeps the order of added callbacks.
    """

    def __init__(self, name="registry"):
        super().__init__(name)
        self.callbacks = OrderedDict()

    def on_epoch_start(self, phase, epoch):
        for c in self.callbacks.values():
            c.on_epoch_start(phase, epoch)

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        for c in self.callbacks.values():
            c.on_step(inputs, outputs, labels, mask, loss, step)

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        for c in self.callbacks.values():
            c.on_epoch_end(epoch)

    def __contains__(self, o):
        return self.callbacks.__contains__(o)

    def __getitem__(self, key):
        return self.callbacks[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Callback):
            raise Exception("Value to add is no Callback, but %s" % value.__class__)
        self.callbacks[key] = value

    def __len__(self):
        return len(self.callbacks)
