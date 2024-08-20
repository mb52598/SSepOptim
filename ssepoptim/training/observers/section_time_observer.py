import logging
from typing import Any

from ssepoptim.utils.context_timer import CtxTimer

from ssepoptim.training.training_observer import TrainingObserver

logger = logging.getLogger(__name__)


class SectionTimeObserver(TrainingObserver):
    def __init__(self):
        self._timer = CtxTimer()

    def _reset_timer(self):
        self._timer.reset()

    def _log_timer(self, prefix: str):
        logger.info("%s|Time: %f s", prefix, self._timer.total)

    def on_training_start(self, locals: dict[str, Any]):
        self._reset_timer()

    def on_training_end(self, locals: dict[str, Any]):
        self._log_timer("Training")

    def on_fine_tuning_start(self, locals: dict[str, Any]):
        self._reset_timer()

    def on_fine_tuning_end(self, locals: dict[str, Any]):
        self._log_timer("Fine-Tune")

    def on_testing_start(self, locals: dict[str, Any]):
        self._reset_timer()

    def on_testing_end(self, locals: dict[str, Any]):
        self._log_timer("Test")
