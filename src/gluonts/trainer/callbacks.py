from collections import defaultdict, deque
import numpy as np
import warnings
import time
from tqdm.auto import tqdm


_TRAIN = 'train'
_TEST = 'test'
_PREDICT = 'predict'


class CallbackFactory:
    """Container abstracting a list of callbacks.
    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None, queue_length=10, verbose=1):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length
        self.params = {}
        self.model = None
        self.epochs = None
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

    def _reset_batch_timing(self):
        self._delta_t_batch = 0.
        self._delta_ts = defaultdict(lambda: deque([], maxlen=self.queue_length))

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def _call_batch_hook(self, mode, hook, batch, logs=None):
        """Helper function for all batch_{begin | end} methods."""
        if not self.callbacks:
            return
        hook_name = 'on_{mode}_batch_{hook}'.format(mode=mode, hook=hook)
        if hook == 'end':
            if not hasattr(self, '_t_enter_batch'):
                self._t_enter_batch = time.time()
            # Batch is ending, calculate batch time
            self._delta_t_batch = time.time() - self._t_enter_batch

        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            batch_hook = getattr(callback, hook_name)
            batch_hook(batch, logs)
        self._delta_ts[hook_name].append(time.time() - t_before_callbacks)

        delta_t_median = np.median(self._delta_ts[hook_name])
        if (self._delta_t_batch > 0. and
           delta_t_median > 0.95 * self._delta_t_batch and
           delta_t_median > 0.1):
            warnings.warn(
                'Method (%s) is slow compared '
                'to the batch update (%f). Check your callbacks.'
                % (hook_name, delta_t_median), RuntimeWarning)

        if hook == 'begin':
            self._t_enter_batch = time.time()

    def on_batch_begin(self, batch, logs=None):
        self._call_batch_hook(_TRAIN, 'begin', batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self._call_batch_hook(_TRAIN, 'end', batch, logs=logs)
        self.pbar_batch.update(1)

        d = dict(zip(*self.params['metrics'].get()))
        self.pbar_batch.set_postfix(**d)

    def on_epoch_begin(self, epoch, logs=None):
        """Calls the `on_epoch_begin` methods of its callbacks.
        This function should only be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
        self._reset_batch_timing()

    def on_epoch_end(self, epoch, logs=None):
        """Calls the `on_epoch_end` methods of its callbacks.
        This function should only be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

        if epoch < self.epochs-1:
            self.pbar_batch.reset()
        self.pbar_epoch.update(1)
        d = dict(zip(*self.params['metrics'].get()))
        self.pbar_epoch.set_postfix(**d)
        self.pbar_epoch.write(f'Epoch [{epoch+1}/{self.epochs}]: {self.pbar_epoch.postfix}')

    def on_train_batch_begin(self, batch, logs=None):
        """Calls the `on_train_batch_begin` methods of its callbacks.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """
        self._call_batch_hook(_TRAIN, 'begin', batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Calls the `on_train_batch_end` methods of its callbacks.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """
        self._call_batch_hook(_TRAIN, 'end', batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Calls the `on_test_batch_begin` methods of its callbacks.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """
        self._call_batch_hook(_TEST, 'begin', batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        """Calls the `on_test_batch_end` methods of its callbacks.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """
        self._call_batch_hook(_TEST, 'end', batch, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        """Calls the `on_predict_batch_begin` methods of its callbacks.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """
        self._call_batch_hook(_PREDICT, 'begin', batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        """Calls the `on_predict_batch_end` methods of its callbacks.
        # Argument
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """
        self._call_batch_hook(_PREDICT, 'end', batch, logs=logs)

    def on_train_begin(self, logs=None):
        """Calls the `on_train_begin` methods of its callbacks.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_train_begin(logs)

        self.epochs = self.params["epochs"]

        self.pbar_epoch = tqdm(
            total=self.epochs,
            desc="Epoch",
            unit="epoch",
            position=0,
            bar_format=self.pbar_fmt,
        )
        self.pbar_batch = tqdm(
            total=self.params["steps"],
            desc="--> Batch",
            unit="batch",
            position=1,
            leave=True,
            bar_format=self.pbar_fmt,
        )

    def on_train_end(self, logs=None):
        """Calls the `on_train_end` methods of its callbacks.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_train_end(logs)

        self.pbar_epoch.close()
        self.pbar_batch.close()

    def on_test_begin(self, logs=None):
        """Calls the `on_test_begin` methods of its callbacks.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        """Calls the `on_test_end` methods of its callbacks.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        """Calls the `on_predict_begin` methods of its callbacks.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        """Calls the `on_predict_end` methods of its callbacks.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


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

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.
        # Arguments
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """
        # For backwards compatibility
        self.on_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """
        # For backwards compatibility
        self.on_batch_end(batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.
        Also called at the beginning of a validation batch in the `fit` methods,
        if validation data is provided.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.
        Also called at the end of a validation batch in the `fit` methods,
        if validation data is provided.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.
        Subclasses should override for any actions to run.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_predict_end(self, logs=None):
        """Called at the end of prediction.
        Subclasses should override for any actions to run.
        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """


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

    def __init__(self, filepath, monitor='loss', verbose=0,
                 save_best_only=True, save_weights_only=False,
                 mode='auto', period=1):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        self.metric_idx = None

    def _get_metric(self):
        names, all_metrics = self.params['metrics'].get()
        return all_metrics[self.metric_idx]

    def on_train_begin(self, logs=None):
        names, all_metrics = self.params['metrics'].get()
        self.metric_idx = names.index(self.monitor)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)

            if self.save_best_only:
                current = self._get_metric()
                if current is None:
                    warnings.warn("Can save best model only with {} available, "
                                  "skipping.".format(self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_parameters(filepath)
                        else:
                            self.model.export(filepath, epoch)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))

                if self.save_weights_only:
                    self.model.save_parameters(filepath)
                else:
                    self.model.export(filepath, epoch)


class History(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in zip(*self.params["metrics"].get()):
            self.history.setdefault(k, []).append(v)


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered.
    """

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = self.params['metrics'].get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.halt = True


# class ProgbarLogger(Callback):
#     """Callback that prints metrics to stdout.
#
#     # Arguments
#         count_mode: One of "steps" or "samples".
#             Whether the progress bar should
#             count samples seen or steps (batches) seen.
#         stateful_metrics: Iterable of string names of metrics that
#             should *not* be averaged over an epoch.
#             Metrics in this list will be logged as-is.
#             All others will be averaged over time (e.g. loss, etc).
#
#     # Raises
#         ValueError: In case of invalid `count_mode`.
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.pbar_epoch = None
#         self.pbar_batch = None
#
#     def on_train_begin(self, logs=None):
#         self.verbose = self.params["verbose"]
#         self.epochs = self.params["epochs"]
#
#         l_bar = "{desc} [{n_fmt}/{total_fmt}] |"
#         r_bar = "| {percentage:3.0f}% [{elapsed_s:3.1f}s < -{remaining_s:3.1f}s, {rate_fmt}{postfix}]"
#         fmt = l_bar + "{bar}" + r_bar
#
#         self.pbar_epoch = tqdm(
#             total=self.params['epochs'],
#             desc='Epoch',
#             unit='epoch',
#             position=0,
#             bar_format=fmt,
#         )
#         self.pbar_batch = tqdm(
#             total=self.params['steps'],
#             desc='Batch',
#             unit='batch',
#             position=1,
#             leave=True,
#             bar_format=fmt,
#         )
#
#     def on_epoch_begin(self, epoch, logs=None):
#         if not self.verbose:
#             return
#         # self.pbar_epoch.set_description('Epoch [{}/{}]'.format(epoch, self.epochs))
#
#     def on_batch_begin(self, batch, logs=None):
#         pass
#
#     def on_batch_end(self, batch, logs=None):
#         self.pbar_batch.update(1)
#
#         d = dict(zip(*self.params['metrics'].get()))
#         self.pbar_batch.set_postfix(**d)
#
#     def on_epoch_end(self, epoch, logs=None):
#         if epoch < self.epochs-1:
#             self.pbar_batch.reset()
#         self.pbar_epoch.update(1)
#         d = dict(zip(*self.params['metrics'].get()))
#         self.pbar_epoch.set_postfix(**d)
#
#         postfix = ""
#         for k, v in d.items():
#             postfix += f"{k}={v:3.2f}, "
#
#         self.pbar_epoch.write(f'Epoch [{epoch+1}/{self.epochs}]: {self.pbar_epoch.postfix}')
#
#     def on_train_end(self, logs=None):
#         self.pbar_batch.close()
#         self.pbar_epoch.close()
