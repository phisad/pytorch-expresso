from collections import OrderedDict

import torch


class Callback(object):
    """
    Base class for callbacks.
    """

    def __init__(self, name: str, on_phase=None):
        # As default, we apply callbacks only during training
        if on_phase is None:
            on_phase = ["train", "validate"]
        self.name = name
        self.on_phase = on_phase
        self.current_phase = None

    def is_applicable(self):
        return self.current_phase in self.on_phase

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if not self.is_applicable():
            return
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


class CallbackRegistry(Callback):
    """
        Register one or more callbacks for callback method invocation. Keeps the order of added callbacks.
    """

    def __init__(self, name="registry", on_phase: list = None):
        super().__init__(name, on_phase)
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

    def __iter__(self):
        return self.callbacks.__iter__()

    def __len__(self):
        return len(self.callbacks)
