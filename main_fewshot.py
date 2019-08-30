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
import collections
from pathlib import Path

import tensorflow as tf
from tensorflow import keras as K

from data_kits import data_factory as factory
from lib import solver
from networks import few_shot as net_lib
from utils import logger as logging
from utils import timer, summary_kits, tools

version = [int(x) for x in tf.__version__.split(".")]
if version[0] == 1 and version[1] >= 14:
    from tensorflow.python.util import deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False

__all_dataset = [
    "omniglot"
]


def get_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Global Arguments")
    group.add_argument("--mode", default="train", type=str, choices=["train", "test", "infer"])
    group.add_argument("--tag", type=str, required=True, help="Model tag")
    group.add_argument("--model_dir", type=str)
    group.add_argument("--log_step", type=int, default=10000000, help="Step interval for logging train info")
    group.add_argument("--log_file", type=str, help="Logging output into")

    group = parser.add_argument_group(title="Dataset Arguments")
    group.add_argument("--dataset", type=str, default="omniglot", choices=__all_dataset, help="The dataset name")
    group.add_argument("-b", "--batch_size", type=int, default=4, help="mini-batch size (default: %(default)d)")
    group.add_argument("--no_val", action="store_true", help="If set, then don't split validation set "
                                                             "from training set.")

    group = parser.add_argument_group(title="Model Arguments")
    group.add_argument("--net_name", type=net_lib.checker,
                       default="matchingnetwork", help="the name for network to use")
    group.add_argument("--init_channel", type=int, help="Output channel of the first conv layer(default: 64 for resnet")
    group.add_argument("-dr", "--drop_rate", type=float, default=0, help='dropout rate')
    group.add_argument("-wd", "--weight_decay", type=float,
                       default=1e-4, help="Weight decay rate. (default %(default)f)")
    group.add_argument("--ckpt", type=str, help="You can specify a checkpoint for restoring. "
                                                "If not specified, the program will try to restore checkpoint "
                                                "from 'model_dir/tag' directory.")
    group.add_argument("-sb", "--save_best_ckpt", action="store_true", help="Save best checkpoint")
    group.add_argument("-lb", "--load_best_ckpt", action="store_true", help="Load best checkpoint")

    group = parser.add_argument_group(title="Few Shot Arguments")
    group.add_argument("-w", "--num_ways", type=int, default=5)
    group.add_argument("-s", "--num_shots", type=int, default=1)
    group.add_argument("--num_train_batches", type=int, default=1000, help="Number of training batches per epoch")
    group.add_argument("--num_val_batches", type=int, default=250, help="Number of validation batches per epoch")
    group.add_argument("--num_test_batches", type=int, default=1000, help="Number of test batches in total")
    group.add_argument("--train_seed", type=int, help="Random seed for generate training batches")
    group.add_argument("--val_seed", type=int, default=1234, help="Random seed for generate validation batches")
    group.add_argument("--test_seed", type=int, default=5678, help="Random seed for generate test batches")
    group.add_argument("--aug_class", action="store_true", help="Augment classes by rotating k * 90 degrees")

    group = parser.add_argument_group(title="MatchingNetwork Arguments")
    group.add_argument("--fce", action="store_true", help="Whether or not to use Fully Context Embeddings module")
    group.add_argument("--process_steps", type=int, default=5, help="Number of processing steps in Fully Context "
                                                                    "Embeddings module")

    solver.add_arguments(parser)
    return parser


def get_model(sup_inputs, sup_labels, que_inputs, que_labels, mode, dataset, args):
    """
    Define core model and loss function using inputs x_images and y_labels

    Parameters
    ----------
    sup_inputs: Tensor
        support set inputs with shape [batch_size, num_ways, num_shots, height, width, channels]
    sup_labels: Tensor
        support set ground truth tensor with shape [batch_size, num_ways, num_shots]
    que_inputs: Tensor
        query image with shape [batch_size, height, width, channels]
    que_labels: Tensor
        query image ground truth with shape [batch_size]
    mode: str
        Training mode. Valid values are [train, test]
    dataset: dict
        dataset object
    args:
        command line arguments

    Returns
    -------
    scaffold: dict
        A scaffold contains fetches, optimizer, metrics, summary writer, saver, etc.
    """
    sup_inputs = K.Input(tensor=sup_inputs)
    sup_labels = K.Input(tensor=sup_labels)
    que_inputs = K.Input(tensor=que_inputs)
    if "matchingnetwork" in args.net_name:
        kwargs = {"kernel_initializer": "he_normal"}
        net = net_lib.MatchingNetwork([64] * 4, dataset["num_classes"], args.process_steps, args.fce,
                                      sup_shape=dataset["shape"], **kwargs)
    else:
        raise NotImplementedError

    y_logits = net((sup_inputs, sup_labels, que_inputs))
    model = K.Model(inputs=[sup_inputs, sup_labels, que_inputs], outputs=y_logits)
    scaffold = {"model": model}

    with tf.name_scope("Loss"):
        que_labels_oh = tf.one_hot(que_labels, args.num_ways)
        ce_loss = -tf.reduce_sum(que_labels_oh * tf.log(y_logits), axis=-1)
        ce_loss = tf.reduce_mean(ce_loss)
        # regu_loss = tf.add_n(model.losses)
        # total_loss = ce_loss + regu_loss
        total_loss = ce_loss

    if mode == "train":
        optimizer = solver.Solver(args, dataset["train"]["steps"])
        scaffold["optimizer"] = optimizer
        scaffold["fetches"] = {"train_op": optimizer.minimize(total_loss, model.updates),
                               # "total_loss": total_loss, "regu_loss": regu_loss}
                               "total_loss": total_loss}
        # Define checkpoint saver and summary writer
        scaffold["writer"] = tf.summary.FileWriter(args.model_dir, graph=tf.get_default_graph())
        # Define summary
        with tf.name_scope("summary_image"):
            patched_row = tf.concat(tf.unstack(sup_inputs, axis=1), axis=2)     # [bs, shots, ways * h, w, 1]
            patched_image = tf.concat(tf.unstack(patched_row, axis=1), axis=2)    # [bs, ways*h, shots * w, 1]
        tf.summary.image("sup_images", patched_image)
        tf.summary.image("que_image", que_inputs)
        tf.summary.scalar("learning_rate", optimizer.lr)
        scaffold["summaries"] = tf.summary.merge_all()
    elif mode == "test":
        scaffold["fetches"] = {"total_loss": total_loss}
    else:
        raise NotImplementedError

    scaffold["saver"] = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    if args.save_best_ckpt:
        scaffold["best_saver"] = tf.train.Saver(saver_def=scaffold["saver"].as_saver_def())

    with tf.name_scope("Metric"):
        y_pred = tf.argmax(y_logits, axis=1, output_type=tf.int32)
        accuracy, acc_update = tf.metrics.accuracy(que_labels, y_pred)
    scaffold["metrics"] = {"acc": accuracy, "acc_up": acc_update}

    return scaffold


def train(args, dataset, scaffold, logger):
    sess_cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=sess_cfg)

    fetches = scaffold["fetches"]
    optimizer = scaffold["optimizer"]
    saver = scaffold["saver"]
    writer = scaffold["writer"]
    metrics = scaffold["metrics"]
    summaries = scaffold["summaries"]
    require_val = not args.no_val

    logger.info(optimizer)

    # After create session
    sess.run(tf.global_variables_initializer())
    logger.info("Global variable initialized")
    local_var_init = tf.local_variables_initializer()
    sess.run(local_var_init)
    logger.info("Local variable initialized")
    # Load checkpoint if possible
    ckpt = tf.train.get_checkpoint_state(args.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        logger.info("Restoring parameters from %s", ckpt.model_checkpoint_path)

    tr_feed_dict = {K.backend.learning_phase(): 1}
    val_feed_dict = {K.backend.learning_phase(): 0}

    if require_val:
        # Get train/val string handler
        train_handler, val_handler = sess.run(
            [dataset["train"]["iter"].string_handle(), dataset["val"]["iter"].string_handle()])
        tr_feed_dict[dataset["handler"]] = train_handler
        val_feed_dict[dataset["handler"]] = val_handler

    finished_epoch = sess.run(optimizer.global_step) // dataset["train"]["steps"]
    total_epochs = args.epochs or args.total_epochs - finished_epoch

    log_loss_acc = tools.Accumulator()
    total_loss_acc = tools.Accumulator()
    # regu_loss_acc = tools.Accumulator()
    val_loss_acc = tools.Accumulator()
    best_acc = 0.
    best_epoch = 0.
    ti = timer.Timer()
    logger.info("Start training ...")

    for i in range(total_epochs):
        # Train
        sess.run(dataset["train"]["iter"].initializer)
        logger.info("Epoch %d/%d - Learning rate: %.4g", i + 1, total_epochs, sess.run(optimizer.lr))
        while True:
            try:
                ti.tic()
                if ti.calls == 0:
                    summary_val, fetches_val = sess.run([summaries, fetches], tr_feed_dict)
                    writer.add_summary(summary_val, global_step=i)  # Here we refer global step to #epoch
                else:
                    fetches_val = sess.run(fetches, tr_feed_dict)
                ti.toc()
                total_loss_acc.update(fetches_val["total_loss"])
                log_loss_acc.update(fetches_val["total_loss"])
                # regu_loss_acc.update(fetches_val["regu_loss"])
                if ti.calls % args.log_step == 0:
                    logger.info("Epoch %d/%d Step %d/%d - Train loss: %.4f - %.2f step/s",
                                i + 1, total_epochs, ti.calls, dataset["train"]["steps"],
                                log_loss_acc.pop(), ti.speed)
            except tf.errors.OutOfRangeError:
                break

        # At epoch end
        val_summ = collections.OrderedDict()
        if require_val:
            sess.run(dataset["val"]["iter"].initializer)
            while True:
                try:
                    ti.tic()
                    total_loss_val, _ = sess.run([fetches["total_loss"], metrics["acc_up"]], val_feed_dict)
                    ti.toc()
                    val_loss_acc.update(total_loss_val)
                except tf.errors.OutOfRangeError:
                    break
            acc_val = sess.run(metrics["acc"])
            if acc_val > best_acc:
                best_acc = acc_val
                best_epoch = i + 1
                if args.save_best_ckpt:
                    save_path = scaffold["best_saver"].save(
                        sess, args.model_dir + "/best_" + args.tag, i, write_meta_graph=False,
                        latest_filename="best_checkpoint")
                    logger.info("Save (best) checkpoint to %s", save_path)
            val_summ["acc"] = acc_val
            val_summ["val_loss"] = val_loss_acc.pop()
            sess.run(local_var_init)  # Reset accuracy local variables 'count' and 'total'
            logger.info("Epoch %d/%d - Train loss: %.4f, Val loss: %.4f, Val acc: %.4f, %.2f step/s",
                        i + 1, total_epochs, total_loss_acc.avg, val_summ["val_loss"],
                        val_summ["acc"], ti.speed)
        else:
            logger.info("Epoch %d/%d - Train loss: %.4f, %.2f step/s",
                        i + 1, total_epochs, total_loss_acc.avg, ti.speed)
        # summary_kits.summary_scalar(writer, i, ["train_loss", "regu_loss"] + list(val_summ.keys()),
        #                             [total_loss_acc.pop(), regu_loss_acc.pop()] + list(val_summ.values()))
        summary_kits.summary_scalar(writer, i, ["train_loss"] + list(val_summ.keys()),
                                    [total_loss_acc.pop()] + list(val_summ.values()))
        save_path = saver.save(sess, args.model_dir + "/" + args.tag, i, write_meta_graph=False)
        logger.info("Save checkpoint to %s", save_path)
        ti.reset()
    logger.info("Best val acc: %.4f in epoch %d.", best_acc, best_epoch)
    sess.close()
    writer.close()


def test(args, dataset, scaffold, logger):
    sess_cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=sess_cfg)

    fetches = scaffold["fetches"]
    metrics = scaffold["metrics"]

    # Load checkpoint if possible
    saver = scaffold["saver"]
    if args.ckpt and Path(args.ckpt + ".index").exists():
        checkpoint_path = args.ckpt
    else:
        latest_filename = "best_checkpoint" if args.load_best_ckpt else None
        ckpt = tf.train.get_checkpoint_state(args.model_dir, latest_filename)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint_path = ckpt.model_checkpoint_path
        else:
            raise ValueError("Missing checkpoint for restoring.")
    saver.restore(sess, checkpoint_path)
    logger.info("Restoring parameters from %s", checkpoint_path)

    test_loss_acc = tools.Accumulator()
    ti = timer.Timer()
    logger.info("Start testing ...")

    # Test
    sess.run([dataset["parent_iter"].initializer, tf.local_variables_initializer()])
    while True:
        try:
            ti.tic()
            total_loss_val, _ = sess.run([fetches["total_loss"], metrics["acc_up"]])
            ti.toc()
            test_loss_acc.update(total_loss_val)
        except tf.errors.OutOfRangeError:
            break
    acc_val = sess.run(metrics["acc"])
    logger.info("Test loss: %.4f, Test acc: %.4f, %.2f step/s", test_loss_acc.avg, acc_val, ti.speed)
    sess.close()


def main():
    parser = get_parser()
    args = parser.parse_args()
    if not args.model_dir:
        model_dir = Path(__file__).parent / "model_dir" / args.tag
        model_dir.mkdir(parents=True, exist_ok=True)
        args.model_dir = str(model_dir)
    logger = logging.tf_logger(logdir=Path(args.model_dir) / "logs",
                               suffix="{}_{}".format(args.tag, args.mode),
                               out_file=args.log_file)
    logger.debug(args)

    graph = tf.Graph()
    with graph.as_default():
        if args.mode == "train":
            # Dataset
            with tf.name_scope("DataLoader"):
                dataset = factory.few_shot_dataset(args.dataset, "train", args.batch_size,
                                                   args.num_ways, args.num_shots, not args.no_val,
                                                   args.num_train_batches, args.num_val_batches, None,
                                                   args.aug_class, args.train_seed, args.val_seed)
                inputs = dataset["parent_iter"].get_next()
            # We call learning_phase() here to create placeholder outside model scope.
            K.backend.learning_phase()
            scaffold = get_model(*inputs, args.mode, dataset, args)
            train(args, dataset, scaffold, logger)
        elif args.mode == "test":
            with tf.name_scope("DataLoader"):
                dataset = factory.few_shot_dataset(args.dataset, "test", args.batch_size,
                                                   args.num_ways, args.num_shots,
                                                   num_test_batches=args.num_test_batches,
                                                   augment_classes=args.aug_class,
                                                   test_seed=args.test_seed)
                inputs = dataset["parent_iter"].get_next()
            scaffold = get_model(*inputs, args.mode, dataset, args)
            test(args, dataset, scaffold, logger)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
