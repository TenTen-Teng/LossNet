from typing import Any, Dict, Optional, TypeVar
import json

# Define types.
TLossNetConfig = TypeVar("TLossNetConfig", bound="LossNetConfig")

class LossNetConfig:
    """Configuration for the LossNet class."""
    
    def __init__(
        self,
        data_format: str = "channels_last",
        intermediate_size: int = 128,
        ndims: int = 3,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        
        :param data_format: The ordering of the dimensions in the inputs. Either
            "channels_last" or "channels_first". "channels_last" corresponds to inputs
            with shape (batch, steps, features) while "channels_fist" corresponds to
            inputs with shape (batch, features, steps).
        :param intermediate_size: Dimension of the fully-connected layers in the LossNet
            model.
        :param ndims: Dimensions of the inputs. "3" as a 3D tensor, "4" as a 4D tensor.
        :param random_seed: Random seed for weights initializer.
        """
        
        assert data_format in ["channels_last", "channels_first"]
        assert isinstance(intermediate_size, int) and intermediate_size > 0
        assert isinstance(ndims, int) and ndims in [3, 4]
        
        self.data_format = data_format
        self.intermediate_size = intermediate_size
        self.ndims = ndims
        self.random_seed = random_seed
        
    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any] = None) -> TLossNetConfig:
        """Constructs a LossNetConfig object from a python dictionary of parameters.
        
        :param model_dict: Contains various (key, value) pairs for the LossNet model
            architecture.
            
        :return: config, LossNetConfig object.
        """
        
        config = LossNetConfig()
        
        for (key, value) in model_dict.items():
            config.__dict__[key] = value
            
        return config
    
    @classmethod
    def from_json_file(cls, json_filepath: str = None) -> TLossNetConfig:
        """Constructs a LossNetConfig object from a JSON file of parameters.
        
        :param json_filepath: File path to a JSON file containing parameters for the
            LossNet model.
            
        :return: LossNetConfig object.
        """
        
        with open(json_filepath, "r") as reader:
            text = reader.read()
            
        return cls.from_dict(json.loads(text))
    