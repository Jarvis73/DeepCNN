# Copyright 2019 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================================

import argparse
import tensorflow as tf
from tensorflow.contrib import opt as contrib_opt


class CustomKeys(object):
    LEARNING_RATE = "learning_rate"
    LR_UPDATE_OPS = "lr_update_ops"


def add_arguments(parser):
    group = parser.add_argument_group(title="Training Arguments")
    group.add_argument("-lr", "--learning_rate",
                       type=float, default=1e-3, help="Base learning rate for model training (default: %(default)f)")
    group.add_argument("--learning_policy", choices=["custom_step", "period_step", "poly", "plateau"],
                       type=str, default="period_step", help="Learning rate policy for training (default: %(default)s)")
    group.add_argument("--epochs",
                       type=int, default=0, help="Number of epochs for training")
    group.add_argument("--total_epochs",
                       type=int, default=100, help="Number of total epochs for training")
    group.add_argument("-lrb", "--lr_decay_boundaries",
                       type=int, nargs="*", help="For \"custom_step\" policy. Use the specified learning rate at the "
                                                 "given boundaries.")
    group.add_argument("-lrv", "--lr_custom_values",
                       type=float, nargs="+", help="For \"custom_step\" policy. Use the specified learning rate at "
                                                   "the given boundaries. Make sure "
                                                   "len(lr_custom_values) - len(lr_decay_boundaries) = 1")
    group.add_argument("-lrs", "--lr_decay_step",
                       type=int, default=1e5, help="For \"period_step\" policy. Decay the base learning rate at a "
                                                   "fixed step (default: %(default)d)")
    group.add_argument("-lrr", "--lr_decay_rate",
                       type=float, default=0.1, help="For \"period_step\" and \"plateau\" policy. Learning rate "
                                                     "decay rate (default: %(default)f)")
    group.add_argument("-lrp", "--lr_power",
                       type=float, default=0.9, help="For \"poly\" policy. Polynomial power (default: %(default)f)")
    group.add_argument("--lr_end",
                       type=float, default=1e-6, help="For \"poly\" and \"plateau\" policy. "
                                                      "The minimal end learning rate (default: %(default)f)")
    group.add_argument("--lr_patience",
                       type=int, default=30, help="For \"plateau\" policy. Learning rate patience for decay "
                                                  "(unit: epoch)")
    group.add_argument("--optimizer", choices=["Adam", "Momentum", "AdamW"],
                       type=str, default="Momentum", help="Optimizer for training (default: %(default)s)")
    group.add_argument("--lr_warm_up",
                       action="store_true", help="Warm up with a low lr to stabilize parameters")
    group.add_argument("--warm_up_epoch",
                       type=int, default=1, help="Number epochs for warming up")
    group.add_argument("--slow_start_lr",
                       type=float, default=1e-4, help="Learning rate employed during slow start")

    group = parser.add_argument_group(title="Optimizer Arguments")
    group.add_argument("--adam_beta1", type=float, help=argparse.SUPPRESS)
    group.add_argument("--adam_beta2", type=float, help=argparse.SUPPRESS)
    group.add_argument("--adam_epsilon", type=float, help=argparse.SUPPRESS)
    group.add_argument("--momentum_momentum", type=float, help=argparse.SUPPRESS)
    group.add_argument("--momentum_use_nesterov", action="store_true", help=argparse.SUPPRESS)
    group.add_argument("--adamw_beta1", type=float, help=argparse.SUPPRESS)
    group.add_argument("--adamw_beta2", type=float, help=argparse.SUPPRESS)
    group.add_argument("--adamw_epsilon", type=float, help=argparse.SUPPRESS)


class Solver(object):
    def __init__(self, args, num_iter_per_epoch, name=None, **kwargs):
        """ Don't create ops/tensors in __init__() """
        self._args = args
        self.name = name or "Optimizer"

        # global step tensor
        # Warning: Don't create global step in __init__() function, but __call__()
        self.global_step = None

        self.learning_policy = self.args.learning_policy
        self.base_learning_rate = self.args.learning_rate
        self.learning_rate_decay_step = None if self.args.lr_decay_step is None else \
            num_iter_per_epoch * self.args.lr_decay_step
        self.learning_rate_decay_rate = self.args.lr_decay_rate
        self.learning_power = self.args.lr_power
        self.end_learning_rate = self.args.lr_end
        self.num_iter_per_epoch = num_iter_per_epoch
        self.learning_rate_decay_boundaries = None if self.args.lr_decay_boundaries is None else \
            [x * num_iter_per_epoch for x in self.args.lr_decay_boundaries]
        self.learning_rate_custom_values = self.args.lr_custom_values
        self.warm_up_epoch = self.args.warm_up_epoch

        # Warm up
        self.slow_start_learning_rate = self.args.slow_start_lr

        self.optimizer = self.args.optimizer.lower()
        self.optimizer_params = {}
        for k, v in vars(args).items():
            if k.startswith(self.optimizer) and v is not None:
                res = k.split("_")
                if res[0] == self.optimizer:
                    val = "_".join(res[1:])
                    self.optimizer_params[val] = v

        self.weight_decay = None
        if hasattr(self._args, "weight_decay"):
            self.weight_decay = self._args.weight_decay
        if self.optimizer == "adamw" and self.weight_decay is None:
            raise ValueError("Solver AdamW require 'weight_decay' argument!")

        self.lr_params = {} if not self.args.lr_warm_up else \
            {"slow_start_step": self.num_iter_per_epoch * self.warm_up_epoch,
             "slow_start_learning_rate": self.slow_start_learning_rate}

    @property
    def args(self):
        return self._args

    def _get_model_learning_rate(self, slow_start_step=0, slow_start_learning_rate=1e-4):
        """
        Gets model's learning rate.

        Computes the model's learning rate for different learning policy.

        Right now, only "custom_step", "period_step", "poly" and "plateau" are supported.

        (1) The learning policy for "period_step" is computed as follows:
        current_learning_rate = base_learning_rate *
        learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)

        (2) The learning policy for "poly" is computed as follows:
        current_learning_rate = base_learning_rate *
        (1 - global_step / training_number_of_steps) ^ learning_power

        Parameters
        ----------
        slow_start_step: int
            Training model with small learning rate for the first few steps.
        slow_start_learning_rate: float
            The learning rate employed during slow start.

        Returns
        -------
        Learning rate for the specified learning policy.

        Raises
        ------
        ValueError: If learning policy is not recognized.
        """
        if self.learning_policy == 'period_step':
            learning_rate = tf.train.exponential_decay(
                self.base_learning_rate,
                self.global_step,
                self.learning_rate_decay_step,
                self.learning_rate_decay_rate,
                staircase=True)
        elif self.learning_policy == "custom_step":
            learning_rate = tf.train.piecewise_constant(
                x=self.global_step,
                boundaries=self.learning_rate_decay_boundaries,
                values=self.learning_rate_custom_values
            )
        elif self.learning_policy == 'poly':
            learning_rate = tf.train.polynomial_decay(
                self.base_learning_rate,
                self.global_step,
                self.args.total_epochs * self.num_iter_per_epoch,
                self.end_learning_rate,
                self.learning_power)
        elif self.learning_policy == "plateau":
            learning_rate, update_lr_op = plateau_decay(
                self.base_learning_rate,
                self.learning_rate_decay_rate,
                self.end_learning_rate)
        else:
            raise ValueError('Not supported learning policy.')

        # Employ small learning rate at the first few steps for warm start.
        if slow_start_step > 0:
            learning_rate = tf.where(self.global_step < slow_start_step, slow_start_learning_rate,
                                     learning_rate)
        tf.add_to_collection(CustomKeys.LEARNING_RATE, learning_rate)
        return learning_rate

    def _get_model_optimizer(self, learning_rate):
        if self.optimizer == "adam":
            optimizer_params = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
            optimizer_params.update(self.optimizer_params)
            optimizer = tf.train.AdamOptimizer(learning_rate, **optimizer_params)
        elif self.optimizer == "momentum":
            optimizer_params = {"momentum": 0.9, "use_nesterov": False}
            optimizer_params.update(self.optimizer_params)
            optimizer = tf.train.MomentumOptimizer(learning_rate, **optimizer_params)
        elif self.optimizer == "adamw":
            optimizer_params = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
            optimizer_params.update(self.optimizer_params)
            optimizer = contrib_opt.AdamWOptimizer(self.weight_decay, learning_rate, **optimizer_params)
        else:
            raise ValueError("Not supported optimizer: " + self.optimizer)
        self.optimizer_params = optimizer_params

        return optimizer

    def minimize(self, loss, update_ops=None):
        # Get global step here.
        # __call__() function will be called inside user-defined graph
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope("Optimizer"):
            self.lr = self._get_model_learning_rate(**self.lr_params)
            optimizer = self._get_model_optimizer(self.lr)

            update_ops = update_ops or tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss, global_step=self.global_step)
            else:
                train_op = optimizer.minimize(loss, global_step=self.global_step)

        return train_op

    def __repr__(self):
        str_ = self.__class__.__name__ + "({opt}:".format(opt=self.optimizer.capitalize())
        for k, v in self.optimizer_params.items():
            str_ += " {}={}".format(k, v)
        str_ += ")"
        return str_


def plateau_decay(lr, factor, min_lr):
    with tf.variable_scope("learning_rate"):
        learning_rate = tf.get_variable("value", dtype=tf.float32,
                                        initializer=lr,
                                        trainable=False,
                                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        update_lr_op = tf.assign(learning_rate, tf.maximum(learning_rate * factor, min_lr))
        tf.add_to_collection(CustomKeys.LR_UPDATE_OPS, update_lr_op)
    return learning_rate, update_lr_op
