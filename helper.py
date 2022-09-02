"""The LossNet model helper functions."""
import tensorflow as tf
import six

from typing import Any, Dict, List, Type, Union

def get_shape_list(
    expected_rank: Union[int, List[int]] = None,
    tensor: tf.Tensor = None,
) -> List[int]:
    """Return a list with the dimensions of the shape of tensor, preferring static
    dimensions.
    
    For example,
        
        x = tf.Variable([[1., 1.]], shape=tf.TensorShape([None, 2]))
        shape = [None, 2]
        dyn_shape = <tf.Tensor: shape=(2,) dtype=int32, numpy=array([1, 2], dtype=int32)> # noqa: E501
        shape = [<tf.Tensor: shape=(), dtype=int32, numpy=1>, 2]
        
    :param expected_rank: Expected rank of tensor. If specified and tensor has a
        different rank, then an exception will be thrown.
    :param tensor: Tensor to find the shape of.
    
    :return: shape, Dimensions of the shape of tensor. All static dimensions will be
        returned as Python integers and dynamic dimensions will be returned as tf.Tensor
        scalars.
    """
    
    # If expected_rank is not NOne, check that the rank of tensor matches expected_rank.
    if expected_rank is not None:
        assert_rank(expected_rank=expected_rank, tensor=tensor)
        
    shape = tensor.shape.as_list()
    non_static_indexes = []
    
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
            
    if not non_static_indexes:
        return shape
    
    dyn_shape = tf.shape(tensor)
    
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
        
    return shape


def assert_rank(
    expected_rank: Union[int, List[int]] = None,
    tensor: Union[tf.Tensor, List[tf.Tensor]] = None,
) -> None:
    """Raises an exception if the rank of tensor does not match expected_rank.
    
    :param expected_rank: Expected rank specified as a Python integer or list of
        integers.
    :param tensor: Tensor or list of tensors to check the rank of.
    
    :raises ValueError: If the expected rank doesn't match the actual rank.
    """
    
    expected_rank_dict = {}
    
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
        
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True
            
    if not isinstance(tensor, list):
        tensor = [tensor]
        
    for _tensor in tensor:
        actual_rank = _tensor.shape.ndims
        
        if actual_rank not in expected_rank_dict:
            raise ValueError(
                "For the tensor, the actual rank '%d' (shape = %s)"
                "is not equal to the expected rank '%s'."
                % (actual_rank, str(_tensor.shape), str(expected_rank))
            )
            
            
def extract_attributes_from_model(
    attributes_list: List[str] = None,
    model: Type = None,
    return_tensor_only: bool = False,
) -> Union[Any, Dict[str, Any]]:
    """Extract the specified attributes from model.
    
    :param attributes_list: Contains a list of attribute names to extract from model.
        These attributes are essentially instance attributes of the model class.
    :param model: A class object representing the model to ectract attributes from.
    :param return_tensor_only: Specifies whether or not to only return the Tensor
        corresponding to the attribute in attributes_list or to return the extracted
        attributes dictionary. If True, then attributes_list can only contain a single
        value.
        
    :return: extracted_attributes_dict or model attribute, Contains (key, value) pairs
        where the keys correspon to the attribute name and the valyes correspond to the
        extracted attributes from the model or just the single attributes from the model.
    :raises AttributeError: If model does not have the __dict__ attribute.
    :raises TypeError: If attribute list is not a list.
    """
    
    assert attributes_list is not None
    
    if not isinstance(attributes_list, list):
        attributes_list = [attributes_list]
        
    if not hasattr(model, "__dict__"):
        raise AttributeError("'model' does not have the '__dict__' attribute!")
        
    for attribute in attributes_list:
        assert isinstance(attribute, str)
        assert attribute in model.__dict__.keys(), (
            "Attribute %s is not in the model!" % attribute
        )
        assert not callable(attribute), (
            "Attribute '%s' is callabel and connot be extracted!" % attribute
        )
        
    if return_tensor_only:
        assert len(attributes_list) == 1, (
            "If return_tensor_only is True, then attribute_list can only contain a "
            "single value!"
        )
        
    extract_attributes_dict = {}
    
    for attribute in attributes_list:
        extract_attributes_dict[attribute] = getattr(model, attribute)
    
    if return_tensor_only:
        return extract_attributes_dict[attributes_list[0]]
    
    return extract_attributes_dict

