
"""MNIST example with LossNet model

This script build a LossNet model with MNIST example in tf2.
"""

import tensorflow as tf
import numpy as np
import os
import horovod.tensorflow as hvd
import argparse

from typing import Any, Dict, Tuple

from helper import extract_attributes_from_model
from lossnet_model import LossNet
from lossnet_prediction_loss import LossNetPredictionLoss
from lossnet_total_loss import LossNetTotalLoss

from mnist_model import MnistModel

# Load data function
def load_data(batch_size: int, use_hvd: bool) -> Tuple[Dict[str, Any], int]:
    """Load MNIST data from tf.keras.datasets. Increase the numer of data by 20 times if
    train model with multi-node in HPC environment.

    :param batch_size: The number of samples per batch
    :param use_hvd: Specifies whether or not to use Horovod (for multi-gpu & multi-node training)


    :returns:
        dataset_dict: The dictionary of dataset for training and test.
        nb_train: The number of training samples.
    """

    # Load MNIST data from tf.keras.datasets.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Or load MNIST from csv files.
#     df_train = pd.read_csv("XXX/mnist_train.csv")
#     df_test = pd.read_csv("XXX/mnist_test.csv")
    
#     x_train = df_train.iloc[:, 1:].values.astype("float32")
#     y_train = df_train.iloc[:, 0].values.astype("int32")
    
#     x_test = df_test.iloc[:, 1:].values.astype("float32")
#     y_test = df_test.iloc[:, 0].values.astype("int32")
    
    
    # Duplicate data 100 times if multi-node training.
    if use_hvd:
        duplicate_times = 20
    else:
        duplicate_times = 1

    nb_train = x_train.shape[0] * duplicate_times
    nb_test = x_test.shape[0] * duplicate_times
    print("%%% nb_train: ", nb_train)

    x_train = np.repeat(x_train, duplicate_times)
    y_train = np.repeat(y_train, duplicate_times)
    x_test = np.repeat(x_test, duplicate_times)
    y_test = np.repeat(y_test, duplicate_times)

    # Prepare training data.
    x_train = x_train.reshape(nb_train, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(nb_test, 28, 28, 1).astype("float32") / 255

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=nb_train).batch(batch_size)

    # Prepare the validation dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    dataset_dict = {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset
    }

    return dataset_dict, nb_train



# Train step function.
@tf.function()
def train_step(x, y, first_batch):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        batch_loss = batch_loss_fn(y, logits)
        scalar_loss = loss_fn(y, logits)
        
        if use_amp and not use_lossnet:
            scalar_loss = optimizer.get_scaled_loss(scalar_loss)

        with tf.GradientTape() as lossnet_tape:
            target_features_dict = extract_attributes_from_model(
                attributes_list=model_dict["features_list"],
                model=model
            )
            lossnet_logits = lossnet(target_features_dict)
            
            lossnet_pred_loss = lossnet_prediction_loss(
                y_true=batch_loss,
                y_pred=lossnet_logits,
            )
            total_loss = lossnet_total_loss(
                y_true=scalar_loss,
                y_pred=lossnet_pred_loss,
            )

            if use_amp:
                total_loss = optimizer.get_scaled_loss(total_loss)

        # Horovod: add Horovod Distributed GradientTape.
        if use_hvd:
            lossnet_tape = hvd.DistributedGradientTape(lossnet_tape)
        
        lossnet_gradients = lossnet_tape.gradient(total_loss, lossnet.trainable_variables)

        if use_amp:
            lossnet_gradients = optimizer.get_unscaled_gradients(lossnet_gradients)

        optimizer.apply_gradients(zip(lossnet_gradients, lossnet.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # Training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if use_hvd and first_batch:
            hvd.broadcast_variables(lossnet.variabels, root_rank=0)
            hvd.broadcast_variables(optmizer.variables(), root_rank=0)

    # Horovod: add Horovod Distributed GradientTape.
    if use_hvd:
        tape = hvd.DistributedGradientTape(tape)

    gradients = tape.gradient(total_loss, model.trainable_variables)

    if use_amp and not use_lossnet:
        gradients = optmizer.get_unscaled_gradients(gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # Training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if use_hvd and first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    train_loss_metric(scalar_loss)
    train_acc_metric.update_state(y, logits)



# Set up global variabels.
output_path = "./outputs"
ckpt_name = "model.ckpt"

batch_size = 128

if __name__ == "main":
    output_path = os.path.join(os.getcwd(), "mnist_lossnet")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    parser = argparse.ArgumentParser(
        description="tf.gradientTape mnist_lossnet example."
    )
    parser.add_argument(
        "-hvd",
        "--hvd",
        type=bool,
        required=True,
    )
    parser.add_argument(
        "-amp",
        "--amp",
        type=bool,
        required=True,
    )
    parser.add_argument(
        "-xla",
        "--xla",
        type=bool,
        required=True
    )
    parser.add_argument(
        "-epochs",
        "--epochs",
        type=int,
        required=True
    )
    parser.add_argument(
        "-lossnet",
        "--lossnet",
        required=True
    )

    args = parser.parse_args()

    use_hvd = args.hvd
    use_amp = args.amp
    use_xla = args.xla
    use_lossnet = args.lossnet

    if use_hvd:
        # Initialize Horovod
        hvd.init()

        # Pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experiemental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    if use_amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mix_float16", loss_scale="dynamic")
        tf.keras.mixed_precision.experiemental.set_policy(policy)
        print("Compute dtype: %s" % policy.compute_dtype)
        print("Compute dtype: %s" % policy.variable_dtype)

    if use_xla:
        if tf.config.list_physical_devices("GPU"):
            tf.config.optimizer.set_jit(True)
        else:
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

    # Load data.
    dataset_dict, nb_train = load_data(batch_size=batch_size, use_hvd=use_hvd)

    # Load model.
    model = MnistModel()
    model.build(input_shape=(batch_size, 28, 28, 1))

    # Set up model dictionary.
    model_dict = {
        "data_format": "channels_last",
        "intermediate_size": 6,
        "features_list": ["conv1_output", "conv2_output"],
        "margin": 1.0,
        "ndims": 4,
        "weight": 0.5
    }

    # Set up losses.
    batch_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    lossnet_prediction_loss = LossNetPredictionLoss(margin=model_dict["margin"])
    lossnet_total_loss = LossNetTotalLoss(weight=model_dict["weight"])

    # Set up optimizer.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001 if not use_hvd else 0.001 * hvd.size(),
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )

    if use_amp:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale="dynamic")

    # Build LossNet model.
    target_features_dict = extract_attributes_from_model(
        attributes_list=model_dict["features_list"],
        model=model
    )

    lossnet = LossNet(target_features_dict=target_features_dict, model_dict=model_dict)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy("train_accuracy")
    train_loss_metric = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)

    for epoch in range(args.epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(dataset_dict["train_dataset"]):
            
            train_step(x_batch_train, y_batch_train, first_batch=step == 0)

            checkpoint.save(file_prefix=os.path.join(output_path, ckpt_name))
        
        print(
            "Epoch {}, Accuracy: {}, Loss: {}".format(
                epoch + 1,
                train_acc_metric.result() * 100,
                train_loss_metric.result(),
            )
        )

        # Reset training metrics at the end of each epoch.
        train_acc_metric.reset_states()
        train_loss_metric.reset_states()
            
        tf.print("--- target_model.trainable_variables: ", model.trainable_variables)
        tf.print("--- lossnet.trainable_variables: ", lossnet.trainable_variables, summarize=500)



