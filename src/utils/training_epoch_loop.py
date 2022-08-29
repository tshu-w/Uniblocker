# HACK: https://github.com/Lightning-AI/lightning/issues/14431
from collections import OrderedDict

from pytorch_lightning.loops.epoch.training_epoch_loop import TrainingEpochLoop
from pytorch_lightning.utilities.fetching import (
    AbstractDataFetcher,
    DataLoaderIterDataFetcher,
)
from pytorch_lightning.utilities.model_helpers import is_overridden


def advance(self, data_fetcher: AbstractDataFetcher) -> None:  # type: ignore[override]
    """Runs a single training batch.

    Raises:
        StopIteration: When the epoch is canceled by the user returning -1
    """
    if self.restarting and self._should_check_val_fx():
        # skip training and run validation in `on_advance_end`
        return
    # we are going to train first so the val loop does not need to restart
    self.val_loop.restarting = False

    if not isinstance(data_fetcher, DataLoaderIterDataFetcher):
        batch_idx = self.batch_idx + 1
        batch = next(data_fetcher)
    else:
        batch_idx, batch = next(data_fetcher)
    self.batch_progress.is_last_batch = data_fetcher.done

    kwargs = self._build_kwargs(OrderedDict(), batch, batch_idx)

    self.batch_progress.increment_ready()

    self.trainer._logger_connector.on_batch_start(batch, batch_idx)

    if batch is None:
        self._warning_cache.warn(
            "train_dataloader yielded None. If this was on purpose, ignore this warning..."
        )
        batch_output = []
    else:
        # hook
        self.trainer._call_callback_hooks("on_batch_start")

        # hook
        self.trainer._call_callback_hooks("on_train_batch_start", batch, batch_idx)
        response = self.trainer._call_lightning_module_hook(
            "on_train_batch_start", batch, batch_idx
        )
        self.trainer._call_strategy_hook("on_train_batch_start", batch, batch_idx)
        if response == -1:
            self.batch_progress.increment_processed()
            raise StopIteration

        self.batch_progress.increment_started()

        with self.trainer.profiler.profile("run_training_batch"):
            batch_output = self.batch_loop.run(kwargs)

    self.batch_progress.increment_processed()

    # update non-plateau LR schedulers
    # update epoch-interval ones only when we are at the end of training epoch
    self.update_lr_schedulers("step", update_plateau_schedulers=False)
    if self._num_ready_batches_reached():
        self.update_lr_schedulers("epoch", update_plateau_schedulers=False)

    batch_end_outputs = self._prepare_outputs_training_batch_end(
        batch_output,
        lightning_module=self.trainer.lightning_module,
        num_optimizers=len(self.trainer.optimizers),
    )

    self.trainer._logger_connector.on_batch_end()

    self.batch_progress.increment_completed()

    if is_overridden("training_epoch_end", self.trainer.lightning_module):
        self._outputs.append(batch_output)

    # -----------------------------------------
    # SAVE METRICS TO LOGGERS AND PROGRESS_BAR
    # -----------------------------------------
    self.trainer._logger_connector.update_train_step_metrics()

    # -----------------------------------------
    # VALIDATE IF NEEDED
    # -----------------------------------------
    should_check_val = self._should_check_val_fx()
    if should_check_val:
        self.trainer.validating = True
        self._run_validation()
        self.trainer.training = True

    self.trainer._call_callback_hooks(
        "on_train_batch_end", batch_end_outputs, batch, batch_idx
    )
    self.trainer._call_lightning_module_hook(
        "on_train_batch_end", batch_end_outputs, batch, batch_idx
    )
    self.trainer._call_callback_hooks("on_batch_end")


def on_advance_end(self) -> None:
    # update plateau LR scheduler after metrics are logged
    self.update_lr_schedulers("step", update_plateau_schedulers=True)

    if not self._should_accumulate():
        # this is increased once per batch disregarding multiple optimizers or tbptt on purpose for loggers
        self._batches_that_stepped += 1
    # this will save based on the `batches_that_stepped` value
    self._save_loggers_on_train_batch_end()

    # if training finished, defer exit to the parent. this assumes there will be enough time in between
    # which might not be the case depending on what's in the `*_epoch_end` hooks
    if not self._is_training_done:
        # if fault tolerant is enabled and process has been notified, exit.
        self.trainer._exit_gracefully_on_signal()


TrainingEpochLoop.advance = advance
TrainingEpochLoop.on_advance_end = on_advance_end
