from datetime import datetime

from torchexpresso.callbacks import Callback
import logging

logger = logging.getLogger(__file__)


class RunningLogger(Callback):

    def __init__(self, name="running_logger", period_in_seconds: int = 5):
        super().__init__(name)
        self.log_period_in_seconds = period_in_seconds
        self.current_phase = None
        self.current_epoch = None
        self.time_last_log = None

    def on_epoch_start(self, phase, epoch):
        # TODO: Log every n seconds only, increase period with epoch (at least once every 5 minutes / 300 seconds)
        self.current_phase = phase
        self.current_epoch = epoch
        self.time_last_log = datetime.now()

    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.should_log():
            # TODO: Maybe use "even nicer" python format with padding
            logger.info("Running \t [phase: %s] \t [epoch: %s] \t [step: %s]",
                        self.current_phase, self.current_epoch, step)
            self.time_last_log = datetime.now()

    def on_epoch_end(self, epoch):
        pass

    def should_log(self):
        time_running = datetime.now() - self.time_last_log
        time_running_seconds = time_running.seconds
        return time_running_seconds > self.log_period_in_seconds


class StepLogger(Callback):

    def __init__(self, name="step_logger", period_in_steps: int = 1):
        super().__init__(name)
        self.period_in_steps = period_in_steps
        self.current_phase = None
        self.current_epoch = None
        self.time_last_log = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        self.current_epoch = epoch

    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.period_in_steps == 1 or step % self.period_in_steps == 0:
            # TODO: Maybe use "even nicer" python format with padding
            logger.info("Running \t [phase: %s] \t [epoch: %s] \t [step: %s]",
                        self.current_phase, self.current_epoch, step)
            self.time_last_log = datetime.now()

    def on_epoch_end(self, epoch):
        pass
