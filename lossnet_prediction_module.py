"""This script defines LossNet prediction module."""    

import tensorflow as tf

from typing import Optional

class LossnetPredictionModule(tf.keras.layers.Layer):
    """Base class for LossNet Prediction Module layer."""
    
    def __init__(
        self,
        data_format: str = "channels_last",
        intermediate_size: int = 128,
        random_seed: Optional[int] = None,
        name: str = "lossnet_pred_module",
        ndims: int = 3
    ) -> None:
        """
        
        :param data_format: The ordering of the dimensions in the inputs. Either
            "channels_last" or "channels_first". "channels_last" corresponds to inputs
            with shape (batch, steps, features) while "channels_fist" corresponds to
            inputs with shape (batch, features, steps).
        :param intermediate_size: Dimension of the fully-connected layers in the LossNet
            model.
        :param name: Name for the LossNet Prediction Module layer.
        :param ndims: Dimensions of the inputs. "3" as a 3D tensor, "4" as a 4D tensor.
        :param random_seed: Random seed for weights initializer.
        """
        
        super(LossnetPredictionModule, self).__init__()
        
        # Construct the Global Average Pooling layer.
        if ndims == 3:
            # For the 3D input tensors.
            self.gap = tf.keras.layers.GlobalAveragePooling1D(
                data_format=data_format, name=f"{name}_fc_gap",
            )
        else:
            # For the 4D input tensors.
            self.gap = tf.keras.layers.GlobalAveragePooling2D(
                data_format=data_format, name=f"{name}_fc_gap",
            )
        
        # Construct the Fully Connected layer.
        weights_initializer = tf.keras.initializers.TruncatedNormal(
            stddev=0.02, seed=random_seed,
        )
        self.fc = tf.keras.layers.Dense(
            units=intermediate_size,
            activation="relu",
            kernel_initializer=weights_initializer,
            name=f"{name}_fc",
        )
        
    def call(self, feature_tensor: tf.Tensor = None) -> tf.Tensor:
        """Create the LossNet Prediction Module layer.
        
        :param feature_tensor: The input feature tensor from target model.
        
        :return: The LossNet Prediction Module output.
        """
        
        x = self.gap(feature_tensor)
        
        return self.fc(x)
    
