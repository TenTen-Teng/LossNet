"""LossNet model class."""

import tensorflow as tf
from helper import get_shape_list
from typing import Any, Dict, List, Union

from lossnet_config import LossNetConfig
from lossnet_prediction_module import LossnetPredictionModule

class LossNet(tf.keras.Model):
    """Base class for the LossNet model."""
    
    def __init__(
        self,
        model_dict: Dict[str, Any] = None,
        target_features_dict: Dict[str, Union[tf.Tensor, List[tf.Tensor]]] = None,
    ) -> None:
        """
        
        :param model_dict: Contains various (key, value) pairs of parameters for the
            LossNet model architecture.
        :param target_feature_dict: Contains (key, value) pairs where the keys
            correspond to the feature name from the target model and the values
            correspond to the extracted features (i.e., tensors) from the target model.
        """
        
        super(LossNet, self).__init__()
        
        assert isinstance(model_dict, dict) and bool(model_dict)
        
        self.config = LossNetConfig.from_dict(model_dict=model_dict)
        
        # Check target_features_dict.
        _target_features_dict = self._check_target_features_dict(
            target_features_dict=target_features_dict
        )
        
        self.lossnet_inputs_dict = {}
        
        for (target_name, target_feature_list) in _target_features_dict.items():
            for target_index in range(len(target_feature_list)):
                # Construct the LossNet Prediction Module layers.
                #
                # self.lossnet_inputs_dict: Dict[str, LossnetPredictionModule].
                self.lossnet_inputs_dict[target_name] = LossnetPredictionModule(
                    data_format=self.config.data_format,
                    intermediate_size=self.config.intermediate_size,
                    ndims=self.config.ndims,
                    random_seed=self.config.random_seed,
                    name=f"lossnet_{target_name}_{target_index}",
                )
                
        # Concatenate LossNet Prediction Module layer.
        self.cat = tf.keras.layers.Concatenate(axis=-1)
        
        # Final fully connected layer.
        weights_initializer = tf.keras.initializers.TruncatedNormal(
            stddev=0.02, seed=self.config.random_seed,
        )
        self.lossnet_final_fc = tf.keras.layers.Dense(
            units=1,
            name="lossnet_logits",
            kernel_initializer=weights_initializer,
            activation=None,
            dtype="float32",
        )
        
    def call(
        self, target_features_dict: Dict[str, Union[tf.Tensor, List[tf.Tensor]]] = None
    ) -> None:
        """Create the base LossNet model.
        
        :param target_features_dict: Contains (key, value) pairs where the keys
            correspond to the feature name from the target model and the values
            correspond to the extracted features (i.e., tensors) from the target model.
            
        :return: The output of LossNet model output tensors.
        """
        
        # Check target_freatures_dict.
        _target_features_dict = self._check_target_features_dict(
            target_features_dict=target_features_dict
        )
        
        # Get LossNet Prediction Module outputs.
        #
        # x: tf.Tensor. The output of LossnetPredictionModule layer with shape as (batch_size, config.intermediate_size) # noqa: E501
        # all_lossnet_features_list: [(batch_size, config.intermediate_size) * N] <--> [(8, 128) * N] # noqa: E501
        # where N is the total number of features in target_features_list.
        all_lossnet_features_list = []
        for (target_name, target_feature_list) in _target_features_dict.items():
            for target_feature in target_feature_list:
                
                x = self.lossnet_inputs_dict[target_name](target_feature)
                all_lossnet_features_list.append(x)
                
        # Get concatenation layer output.
        #
        # self.cat_output: (batch_size, config.intermediate_size * N) <--> (32, 128 * 5 = 640) # noqa: E501
        self.cat_output = self.cat(all_lossnet_features_list)
        
        # The output of LossNet model.
        #
        # The shape of output for LossNet model: (batch_size, 1) <--> (32, 1)
        return self.lossnet_final_fc(self.cat_output)
    
    
    def _check_target_features_dict(
        self, target_features_dict: Dict[str, Union[tf.Tensor, List[tf.Tensor]]],
    ) -> Dict[str, List[tf.Tensor]]:
        """Check target festures dictionary.
        
        :param target_features_dict: Contains (key, value) pairs where the keys
            correspond to the feature name from the target model and the values
            correspond to the extracted features (i.e., tensors) from the targe
            model.
            
        :return: target_features_dict, The modified target feartures dictionary.
        """
        
        assert isinstance(target_features_dict, dict) and bool(target_features_dict)
        
        for (target_name, target_feature_list) in target_features_dict.items():
            assert isinstance(target_name, str)
            
            if not isinstance(target_feature_list, list):
                target_features_dict[target_name] = [target_feature_list]
                
            for _target_feature in target_features_dict[target_name]:
                target_shape_list = get_shape_list(
                    expected_rank=[3, 4], tensor=_target_feature
                )
            
                if len(target_shape_list) == 4:
                    if self.config.data_format == "channels_first":
                        assert target_shape_list[2] == target_shape_list[3], (
                            "If data_format is 'channels_first' (batch_size, channels, height, width), " # noqa: E501
                            "then the height and width dimensions must match for "
                            "LossNet global average pooling operation!"
                        )
                    
                    else:
                        assert target_shape_list[1] == target_shape_list[2], (
                            "If data_format is 'channels_last' (batch_size, height, width, channels), " # noqa: E501
                            "then the height and width dimensions must match for "
                            "LossNet global average pooling operation!"
                        )
                        
        return target_features_dict
