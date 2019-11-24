from collections import defaultdict, deque
import numpy as np
import warnings
import time
from tqdm.auto import tqdm
from typing import Optional, Union

from gluonts.core.component import validated


class Callback:
    """
    Base class for all callbacks.

    Properties
    ----------
    params: dict
        Training parameters (eg. verbosity, batch size, number of epochs...).

    model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:
        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    @validated()
    def __init__(self):
        self.model = None
        self.trainer = None
        self.epochs = None
        self._metrics = None
        self.steps = None
        self.batch_size = None

    def initialize(
            self,
            model,
            trainer,
            epochs,
            metrics,
            steps,
            batch_size
    ) -> None:
        self.model = model
        self.trainer = trainer
        self.epochs = epochs
        self._metrics = metrics
        self.steps = steps
        self.batch_size = batch_size

    def on_batch_begin(self, batch):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    def on_batch_end(self, batch):
        """A backwards compatibility alias for `on_train_batch_end`."""

    def on_epoch_begin(self, epoch):
        """Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_epoch_end(self, epoch):
        """Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """

    def on_train_begin(self):
        """Called at the beginning of training.
        Subclasses should override for any actions to run.
        """

    def on_train_end(self):
        """Called at the end of training.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    @property
    def metrics(self):
        if self._metrics is None:
            return
        return dict(zip(*self._metrics.get()))


class CallbackMessage:

    def __init__(self):
        self._messages = []
        self._indent = None

    def add(self, message: Union[str, 'CallbackMessage'], callback: Optional[Callback] = None):
        if isinstance(message, CallbackMessage):
            for msg in message:
                self.add(msg)
        else:
            if callback is not None:
                message = f"({callback.__class__.__name__}) {message}"
            self._messages.append(message)

    @property
    def indent(self):
        return self._indent

    @indent.setter
    def indent(self, value):
        self._indent = value

    def __iter__(self):
        return iter(self._messages)

    def __len__(self):
        return len(self._messages)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if len(self._messages) == 0:
            return ""

        if self.indent:
            messages = [" "*self.indent + msg for msg in self]
        else:
            messages = self._messages

        return "\n".join(messages)

    @property
    def messages(self):
        return self._messages


class CallbackFactory(Callback):
    """Container abstracting a list of callbacks.
    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    @validated()
    def __init__(
            self,
            callbacks: Optional[list] = None,
            queue_length: Optional[int] = 10,
            verbose: Optional[int] = 1,
    ) -> None:
        super().__init__()

        if callbacks is not None:
            assert (
                all(isinstance(cb, Callback) for cb in callbacks)
            ), "All callbacks should be subclasses of the base class `Callback`."
        else:
            callbacks = []

        assert queue_length >= 1, "`queue_length` should be a positive number."
        assert 0 <= verbose <= 2, "`verbose` should be 0, 1, or 2."

        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length
        self.verbose = verbose
        self._reset_batch_timing()

        # specifications for progress bars
        l_bar = "{desc} [{n_fmt}/{total_fmt}] |"
        r_bar = (
            "| {percentage:3.0f}% [{elapsed_s:3.1f}s < "
            "-{remaining_s:3.1f}s, {rate_fmt}{postfix}]"
        )
        self.pbar_fmt = l_bar + "{bar}" + r_bar
        self.pbar_epoch = None
        self.pbar_batch = None

    def _reset_batch_timing(self) -> None:
        self._delta_t_batch = 0.
        self._delta_ts = defaultdict(lambda: deque([], maxlen=self.queue_length))

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def initialize(
            self,
            model,
            trainer,
            epochs,
            metrics,
            steps,
            batch_size
    ) -> None:
        super().initialize(model, trainer, epochs, metrics, steps, batch_size)
        for callback in self.callbacks:
            callback.initialize(model, trainer, epochs, metrics, steps, batch_size)

    def _call_batch_hook(self, hook: str, batch: int) -> CallbackMessage:
        """Helper function for all batch_{begin | end} methods."""
        msg = CallbackMessage()

        if not self.callbacks:
            return msg

        hook_name = "on_batch_{hook}".format(hook=hook)

        if hook == "end":
            if not hasattr(self, "_t_enter_batch"):
                self._t_enter_batch = time.time()
            # Batch is ending, calculate batch time
            self._delta_t_batch = time.time() - self._t_enter_batch

        t_before_callbacks = time.time()

        for i, callback in enumerate(self.callbacks):
            batch_hook = getattr(callback, hook_name)
            cb_msg = batch_hook(batch)
            if cb_msg:
                msg.add(cb_msg, callback)

        self._delta_ts[hook_name].append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts[hook_name])

        if (
            self._delta_t_batch > 0.
            and delta_t_median > 0.95 * self._delta_t_batch
            and delta_t_median > 0.1
        ):
            msg.add(
                "WARNING: Method ({}) is slow compared to the batch update "
                "({:.3f}s). Check your callbacks.".format(
                    hook_name, delta_t_median
                ),
                self
            )

        if hook == "begin":
            self._t_enter_batch = time.time()

        return msg

    def on_batch_begin(self, batch):
        self._call_batch_hook("begin", batch)

    def on_batch_end(self, batch):
        messages = self._call_batch_hook("end", batch) or {}

        if self.verbose:
            self.pbar_batch.update(1)
            self.pbar_batch.set_postfix(**self.metrics)

            if messages:
                batch_descr = f"--> Batch [{batch}/{self.steps}]: "
                self.pbar_batch.write(batch_descr)
                messages.indent = 6
                self.pbar_batch.write(str(messages))

    def on_epoch_begin(self, epoch):
        """
        Calls the `on_epoch_begin` methods of its callbacks.
        This function should only be called during train mode.

        Parameters
        ----------
        epoch: int
            Index of epoch.
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)
        self._reset_batch_timing()

    def on_epoch_end(self, epoch):
        """Calls the `on_epoch_end` methods of its callbacks.
        This function should only be called during train mode.

        Parameters
        ----------
        epoch: int
            Index of epoch.
        """
        messages = CallbackMessage()
        for callback in self.callbacks:
            cb_msg = callback.on_epoch_end(epoch)
            if cb_msg:
                messages.add(cb_msg, callback)

        if self.verbose:
            if epoch < self.epochs - 1:
                self.pbar_batch.reset()

            epoch_descr = f"Epoch [{epoch}/{self.epochs}]: "
            self.pbar_epoch.update(1)
            self.pbar_epoch.set_postfix(**self.metrics)
            self.pbar_epoch.write(f"{epoch_descr}{self.pbar_epoch.postfix}")

            if messages:
                messages.indent = len(epoch_descr)
                self.pbar_epoch.write(str(messages))

    def on_train_begin(self):
        """
        Calls the `on_train_begin` methods of its callbacks.
        """
        for callback in self.callbacks:
            callback.on_train_begin()

        self.pbar_epoch = tqdm(
            total=self.epochs,
            desc="Epoch",
            unit="epoch",
            position=0,
            bar_format=self.pbar_fmt,
        )
        self.pbar_batch = tqdm(
            total=self.steps,
            desc="--> Batch",
            unit="batch",
            position=1,
            leave=True,
            bar_format=self.pbar_fmt,
        )

    def on_train_end(self):
        """Calls the `on_train_end` methods of its callbacks.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_train_end()

        self.pbar_epoch.close()
        self.pbar_batch.close()


class ModelCheckpoint(Callback):
    """
    Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    Parameters
    ----------
    filepath: string,
        path to save the model file.

    monitor: quantity to monitor.

    verbose: verbosity mode, 0 or 1.

    save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.

    save_weights_only: bool, optional
        If True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).

    mode: str, optional
     one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.

    period: Interval (number of epochs) between checkpoints.
    """

    @validated()
    def __init__(
            self,
            base_path: str,
            monitor: Optional[str] = "loss",
            verbose: Optional[int] = 0,
            save_best_only: Optional[bool] = True,
            save_weights_only: Optional[bool] = False,
            rollback_on_lr_change: Optional[bool] = True,
            mode: Optional[str] = "auto",
            period: Optional[int] = 1,
    ):
        super().__init__()

        assert (
            mode in ["auto", "min", "max"]
        ), "ModelCheckpoint mode {} is unknown. Allowed modes: [auto, min, max]."
        assert period > 0, "ModelCheckpoint period should be strictly positive."

        self.monitor = monitor
        self.verbose = verbose
        self.base_path = base_path
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.rollback_on_lr_change = rollback_on_lr_change
        self.current_lr = None
        self.best_epoch = None

        if mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf

    def on_train_begin(self):
        monitored_metrics_names = list(self.metrics.keys())
        assert (
                self.monitor in monitored_metrics_names
        ), "`monitor` should be one of monitored keys {}. Given value: {}.".format(
            monitored_metrics_names,
            self.monitor
        )

        self.best_epoch = None

    def on_epoch_end(self, epoch):
        msg = CallbackMessage()

        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = "{}-{:05d}".format(self.base_path, epoch)

            if self.save_best_only:
                current = self.metrics[self.monitor]

                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        msg.add(
                            "{} improved from {:.5f} to {:.5f}. Checkpoint saved: {}.".format(
                                self.monitor, self.best, current, filepath
                            ),
                            self
                        )
                    self.best = current
                    self.best_epoch = epoch

                    if self.save_weights_only:
                        self.model.save_parameters(filepath)
                    else:
                        self.model.export(filepath, epoch)
            else:
                if self.verbose > 0:
                    msg.add("Checkpoint saved: {}".format(filepath), self)

                current = self.metrics[self.monitor]
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.best_epoch = epoch

                if self.save_weights_only:
                    self.model.save_parameters(filepath)
                else:
                    self.model.export(filepath, epoch)

        if (
            self.rollback_on_lr_change
            and self.current_lr != self.trainer.learning_rate
            and self.best_epoch is not None
        ):
            filepath = "{}-{:05d}".format(self.base_path, self.best_epoch)
            self.model.load_parameters(filepath)
            msg.add(
                "Learning rate changed from {} to {}. Rolled back model to epoch {}.".format(
                    self.current_lr,
                    self.trainer.learning_rate,
                    self.best_epoch
                ),
                self
            )

        self.current_lr = self.trainer.learning_rate
        return msg


class History(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def on_train_begin(self):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch) -> Optional[dict]:
        self.epoch.append(epoch)
        for k, v in self.metrics.items():
            self.history.setdefault(k, []).append(v)


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered.
    """

    def on_batch_end(self, batch):
        msg = CallbackMessage()
        loss = self.metrics["loss"]
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                msg.add("Invalid loss, terminating training.", self)
                self.model.halt = True
        return msg


class MetricAttentiveScheduler(Callback):
    def __init__(
        self,
        mode: str,
        patience: int,
        monitor: str = "loss",
        decay_factor: float = 0.5,
        min_lr: float = 0.0,
    ) -> None:

        super().__init__()

        assert (
            0 < decay_factor < 1
        ), f"decay_factor factor should be between 0 and 1, got {decay_factor}"

        assert patience >= 0, f"patience should be nonnegative, got {patience}"

        assert mode in [
            "min",
            "max",
        ], f"objective should be 'min' or 'max', got {mode}"

        self.decay_factor = decay_factor
        self.patience = patience
        self.mode = mode
        self.min_lr = min_lr
        self.best_epoch = 0
        self.monitor = monitor

        if mode == "min":
            self.best_metric = np.Inf
            self.monitor_op = np.less
        else:
            self.best_metric = -np.Inf
            self.monitor_op = np.greater

    def on_train_begin(self):
        self.best_epoch = 0

    def on_epoch_end(self, epoch):
        msg = CallbackMessage()
        curr_metric = self.metrics[self.monitor]

        if self.monitor_op(curr_metric, self.best_metric):
            self.best_metric = curr_metric
            self.best_epoch = epoch

        if epoch - self.best_epoch >= self.patience:
            lr = max(self.min_lr, self.decay_factor * self.trainer.learning_rate)
            self.trainer.set_learning_rate(lr)
            msg.add("New learning rate: {}".format(lr), self)
