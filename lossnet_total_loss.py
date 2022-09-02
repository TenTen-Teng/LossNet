"""LossNet Total Loss Class."""

import tensorflow as tf
from typing import Union

class LossNetTotalLoss(tf.keras.losses.Loss):
    """This class creates the LossNet total loss by subclassing from
    tf.keras.losses.Loss.
    """
    
    def __init__(self, weight: float = 0.5, name: str = "lossnet_total_loss") -> None:
        """
        
        :param weight: Scaling constant for weighting the LossNet model loss against the
            target model loss when constructing the overall loss.
        :param name: Name for the LossNet model total loss operation.
        """
        
        super().__init__(name=name)
        self.weight = weight
        
    def call(
        self,
        y_true: Union[None, tf.Tensor] = None,
        y_pred: Union[None, tf.Tensor] = None,
    ) -> tf.Tensor:
        """Contains the logic for loss calculation using y_true, y_pred.
        
        :param y_true: Contains the scalar target model loss.
        :param y_pred: Contains the scalar lossnet model loss.
        
        :return: lossnet_total_loss, The sum of target model loss and lossnet model
            loss.
        """
        
        lossnet_total_loss = tf.math.add(x=y_true, y=self.weight * y_pred)
        
        return lossnet_total_loss
