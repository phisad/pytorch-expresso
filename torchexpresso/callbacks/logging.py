from datetime import datetime

from torchexpresso.callbacks import Callback
import logging

logger = logging.getLogger(__file__)


class RunningLogger(Callback):

    def __init__(self, period_in_seconds: int = 2):
        self.log_period_in_seconds = period_in_seconds
        self.time_epoch_start = None
        self.current_phase = None
        self.current_epoch = None

    def on_epoch_start(self, phase, epoch):
        # TODO: Log every n seconds only, increase period with epoch (at least once every 5 minutes / 300 seconds)
        self.current_phase = phase
        self.current_epoch = epoch
        self.time_epoch_start = datetime.now()

    def on_step(self, inputs, outputs, labels, mask, loss, step):
        time_running = self.time_epoch_start - datetime.now()
        if time_running.seconds % self.log_period_in_seconds:
            # TODO: Maybe use "even nicer" python format with padding
            logger.info("Running \t [phase: %s] \t [epoch: %s] \t [step: %s]",
                        self.current_phase, self.current_epoch, step)

    def on_epoch_end(self, epoch):
        pass
