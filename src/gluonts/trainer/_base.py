# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
import logging
import os
import tempfile
import time
import uuid
from typing import Any, List, NamedTuple, Optional, Union

# Third-party imports
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.gluon.nn as nn
import numpy as np

# First-party imports
from gluonts.core.component import get_mxnet_context, validated
from gluonts.core.exception import GluonTSDataError
from gluonts.dataset.loader import TrainDataLoader, InferenceDataLoader, BatchBuffer
from gluonts.support.util import HybridContext

# Relative imports
from . import learning_rate_scheduler as lrs
from . import callbacks as cb
from .callbacks import Callback

# logger = logging.getLogger("trainer")

# MODEL_ARTIFACT_FILE_NAME = "model"
# STATE_ARTIFACT_FILE_NAME = "state"

# make the IDE happy: mx.py does not explicitly import autograd
mx.autograd = autograd


# def check_loss_finite(val: float) -> None:
#     if not np.isfinite(val):
#         raise GluonTSDataError(
#             "Encountered invalid loss value! Try reducing the learning rate "
#             "or try a different likelihood."
#         )
#
#
# def loss_value(loss: mx.metric.CompositeEvalMetric) -> float:
#     names, metrics = loss.get()
#     return metrics[names.index("loss")]
#     # return loss.get_name_value()[0][1]


# class BestEpochInfo(NamedTuple):
#     params_path: str
#     epoch_no: int
#     metric_value: float


class Trainer:
    r"""
    A trainer specifies how a network is going to be trained.

    A trainer is mainly defined by two sets of parameters. The first one determines the number of examples
    that the network will be trained on (`epochs`, `num_batches_per_epoch` and `batch_size`), while the
    second one specifies how the gradient updates are performed (`learning_rate`, `learning_rate_decay_factor`,
    `patience`, `minimum_learning_rate`, `clip_gradient` and `weight_decay`).

    Parameters
    ----------
    ctx
    epochs
        Number of epochs that the network will train (default: 1).
    batch_size
        Number of examples in each batch (default: 32).
    num_batches_per_epoch
        Number of batches at each epoch (default: 100).
    verbose
        Show information
    learning_rate
        Initial learning rate (default: :math:`10^{-3}`).
    learning_rate_decay_factor
        Factor (between 0 and 1) by which to decrease the learning rate (default: 0.5).
    patience
        The patience to observe before reducing the learning rate, nonnegative integer (default: 10).
    minimum_learning_rate
        Lower bound for the learning rate (default: :math:`5\cdot 10^{-5}`).
    clip_gradient
        Maximum value of gradient. The gradient is clipped if it is too large (default: 10).
    weight_decay
        The weight decay (or L2 regularization) coefficient. Modifies objective by adding a penalty for having
        large weights (default :math:`10^{-8}`).
    init
        Initializer of the weights of the network (default: "xavier").
    hybridize
    """

    @validated()
    def __init__(
        self,
        ctx: Optional[mx.Context] = None,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        verbose: int = 1,
        metrics: Optional[list] = None,
        callbacks: Optional[list] = None,
        learning_rate: float = 1e-3,
        # learning_rate_decay_factor: float = 0.5,
        # patience: int = 10,
        # minimum_learning_rate: float = 5e-5,
        clip_gradient: float = 10.0,
        weight_decay: float = 1e-8,
        init: Union[str, mx.initializer.Initializer] = "xavier",
        hybridize: bool = True,
        optimizer: Optional[mx.optimizer.Optimizer] = None,
        hierarchy_penalty: Optional[float] = None,
    ) -> None:

        assert (
            0 <= epochs < float("inf")
        ), "The value of `epochs` should be >= 0"
        assert 0 < batch_size, "The value of `batch_size` should be > 0"
        assert (
            0 < num_batches_per_epoch
        ), "The value of `num_batches_per_epoch` should be > 0"
        assert (
            0 < learning_rate < float("inf")
        ), "The value of `learning_rate` should be > 0"
        # assert (
        #     0 <= learning_rate_decay_factor < 1
        # ), "The value of `learning_rate_decay_factor` should be in the [0, 1) range"
        # assert 0 <= patience, "The value of `patience` should be >= 0"
        # assert (
        #     0 <= minimum_learning_rate
        # ), "The value of `minimum_learning_rate` should be >= 0"
        assert 0 < clip_gradient, "The value of `clip_gradient` should be > 0"
        assert 0 <= weight_decay, "The value of `weight_decay` should be => 0"

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        # self.learning_rate_decay_factor = learning_rate_decay_factor
        # self.patience = patience
        # self.minimum_learning_rate = minimum_learning_rate
        self.clip_gradient = clip_gradient
        self.weight_decay = weight_decay
        self.init = init
        self.hybridize = hybridize
        self.ctx = ctx if ctx is not None else get_mxnet_context()
        self.halt = False

        self.verbose = verbose
        self.metrics = mx.metric.CompositeEvalMetric((metrics or []) + [mx.metric.Loss()])
        self.hierarchy_penalty = hierarchy_penalty

        if optimizer is None:
            self.optimizer = mx.optimizer.Adam(
                learning_rate=self.learning_rate,
                wd=self.weight_decay,
                clip_gradient=self.clip_gradient,
            )
        else:
            self.optimizer = optimizer

        # Initialize callbacks
        callbacks = callbacks or []
        self.history = cb.History()
        callbacks.append(self.history)
        self.callbacks = cb.CallbackFactory(callbacks, verbose=verbose)

    def set_halt(self, signum: int, stack_frame: Any) -> None:
        logging.info("Received signal: {}".format(signum))
        self.halt = True

    def count_model_params(self, net: nn.HybridBlock) -> int:
        params = net.collect_params()
        num_params = 0
        for p in params:
            v = params[p]
            num_params += np.prod(v.shape)
        return num_params

    def __call__(
        self,
        net: nn.HybridBlock,
        input_names: List[str],
        train_iter: TrainDataLoader,
        predict_iter: Optional[list] = None,
    ) -> Callback:  # TODO: we may want to return some training information here
        self.halt = False

        net.initialize(ctx=self.ctx, init=self.init)

        # from supplymodels import utils
        # states = utils.read_json('s3://dibgerge/experiments-6/JL-TI-R/model-36/states.json')

        with HybridContext(
            net=net,
            hybridize=self.hybridize,
            static_alloc=True,
            static_shape=True,
        ):
            batch_size = train_iter.batch_size

            trainer = mx.gluon.Trainer(
                net.collect_params(),
                optimizer=self.optimizer,
                kvstore="device",  # FIXME: initialize properly
            )

            self.callbacks.initialize(
                net,
                trainer,
                self.epochs,
                self.metrics,
                self.num_batches_per_epoch,
                self.batch_size
            )
            self.callbacks.on_train_begin()

            for epoch_no in range(1, 1 + self.epochs):
                self.callbacks.on_epoch_begin(epoch_no)

                if hasattr(net, 'halt') and net.halt:
                    # logging.info(
                    #     f"Epoch[{epoch_no}] Interrupting training"
                    # )
                    break

                self.metrics.reset()

                for batch_no, data_entry in enumerate(train_iter, start=1):
                    self.callbacks.on_batch_begin(batch_no)
                    if hasattr(net, 'halt') and net.halt:
                        break

                    inputs = [data_entry[k] for k in input_names]

                    with mx.autograd.record():
                        print(inputs)
                        output = net(*inputs)

                        # network can returns several outputs, the first being always the loss
                        # when having multiple outputs, the forward returns a list in
                        # the case of hybrid and a tuple otherwise
                        # we may wrap network outputs in the future to avoid this type check
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output

                        # Add hierarchical penalty here is specified
                        if predict_iter is not None and self.hierarchy_penalty is not None:
                            buffer = BatchBuffer(
                                len(predict_iter),
                                self.ctx,
                                train_iter.dtype
                            )

                            # Id's of selected training data in current batch
                            ids = data_entry['source'][1]
                            hierarchy = {}
                            buffer_idx = 0
                            for i, id_ in enumerate(ids):
                                parent = predict_iter[id_ - 1]
                                buffer.add(parent)
                                hierarchy[buffer_idx] = []
                                parent_idx = buffer_idx
                                buffer_idx += 1

                                for v in parent['children'].values():
                                    curr_list = []
                                    for j, vi in enumerate(v):
                                        buffer.add(predict_iter[vi])
                                        curr_list.append(buffer_idx)
                                        buffer_idx += 1
                                    hierarchy[parent_idx].append(curr_list)

                            batch = buffer.next_batch()
                            inputs = [
                                batch[k] if batch[k].shape[1] > 0
                                else None for k in input_names
                            ]
                            output = net(*inputs)

                            penalty = mx.nd.zeros(shape=output.shape[1:], ctx=self.ctx)
                            tot = 0
                            for parent_idx, h in hierarchy.items():
                                parent = output[parent_idx]
                                for kid in h:
                                    kids_sum = mx.nd.zeros(shape=output.shape[1:], ctx=self.ctx)
                                    tot += 1
                                    for kid_idx in kid:
                                        kids_sum = kids_sum + output[kid_idx]

                                    penalty = penalty + (parent - kids_sum)**2
                            penalty = penalty / tot
                            loss = loss.mean() + self.hierarchy_penalty * penalty.mean()
                            step = 1
                        else:
                            step = batch_size

                    loss.backward()
                    trainer.step(step)
                    self.metrics.update(labels=None, preds=loss)

                    # print out parameters of the network at the first pass
                    if batch_no == 2 and epoch_no == 1:
                        net_name = type(net).__name__
                        num_model_param = self.count_model_params(net)
                        logging.info(
                            f"Number of parameters in {net_name}: {num_model_param}"
                        )

                    self.callbacks.on_batch_end(batch_no)

                self.callbacks.on_epoch_end(epoch_no)

            self.callbacks.on_train_end()
            return self.history
