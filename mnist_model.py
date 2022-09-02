"""This script create a 2-layer convolution network model.

This model can be used to do a quick test.
"""

import tensorflow as tf

class MnistModel(tf.keras.Model):
    def __init__(self) -> None:
        super(MnistModel, self).__init__()

        if tf.config.list_physical_devices("GPU"):
            print("The model will run with 4096 units on a GPU.")
            num_units = 4096
        else:
            print("The model will run with 64 units on a GPU.")
            num_units = 64

        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_units,
            kernel_size=2,
            activation="relu",
            name="conv1",
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_units,
            kernel_size=3,
            activation="relu",
            name="conv2",
        )

        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")

        self.logits = tf.keras.layers.Dense(
            units=10,
            activation=None,
            name="outputs",         
        )

    def call(self, inputs):
        self.conv1_out = self.conv1(inputs)
        self.conv2_out = self.conv2(self.conv1_out)
        self.gap_output = self.gap(self.conv2_out)
        self.logits_out = self.logits(self.gap_output)

        out_out = self.out(self.logits_out)
        return out_out