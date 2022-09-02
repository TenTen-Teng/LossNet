"""LossNet Prediction Loss Class."""

import tensorflow as tf
from typing import Union

class LossNetPredictionLoss(tf.keras.losses.Loss):
    """This class creates the LossNet prediction loss by subclassing from
    tf.keras.losses.Loss.
    """
    
    def __init__(
        self, margin: float = 1.0, name: str = "lossnet_prediction_loss"
    ) -> None:
        """
        
        :param margin: Margin of error for the LossNet model loss.
        :param name: Name for the LossNet model loss prediction loss operation.
        """
        
        super().__init__(name=name)
        self.margin = margin
        
    def call(
        self,
        y_true: Union[None, tf.Tensor] = None,
        y_pred: Union[None, tf.Tensor] = None,
    ) -> tf.Tensor:
        """Contains the logic for loss calculation using y_true, y_pred.
        
        :param y_true: Batch loss for target model. The shape of y_true should be
            (batch_size, ) <--> (32, ).
        :param y_pred: The outputs of LossNet model. The shape of lossnet_pred_logits
            is (batch_size, 1) <--> (32, 1).
            
        :return: lossnet_pred_loss, The LossNet prediction loss value.
        """
        
        assert (
            y_true.shape[0] % 2 == 0
        ), "LossNet prediction loss requires the numerb of batch is even!"
        
        # Remove singleton dimension.
        #
        # logits: (batch_size, ) <--> (32, )
        logits = tf.squeeze(input=y_pred, axis=-1)
        
        # Compute the difference between random paris of the predicted losses.
        #
        # diff_x1_x2: (batch_size // 2, ) <--> (16, )
        x2 = tf.reverse(logits, axis=[0])
        diff_x1_x2 = logits - x2
        diff_x1_x2 = diff_x1_x2[: tf.shape(diff_x1_x2)[0] // 2]
        
        # Compute the difference between random pairs of the target losses.
        #
        # diff_x1_x2: (batch_size // 2, ) <--> (16, )
        y2 = tf.reverse(y_true, axis=[0])
        diff_y1_y2 = y_true - y2
        diff_y1_y2 = diff_y1_y2[: tf.shape(diff_y1_y2)[0] // 2]
        
        # Margin ranking loss.
        #
        # loss(x, y) = max(0, -y * (x1 - x2) + margin)
        # If y = 1 then it assumed the first input should be ranked higher (i.e., have a
        # larger value) than the second inout and vice-versa for y = -1.
        #
        # loss: (batch_size // 2, ) <--> (16, )
        y = -tf.math.sign(diff_y1_y2)
        _y = tf.math.multiply(y, diff_x1_x2) + self.margin
        
        lossnet_pred_loss = tf.math.maximum(0.0, _y)
        
        # Average lossnet_pred_loss for the mini-batch.
        #
        # lossnet_pred_loss: Scalar
        lossnet_pred_loss = tf.math.reduce_mean(lossnet_pred_loss)
        
        return lossnet_pred_loss
    
    