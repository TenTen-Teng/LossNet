# LossNet Model - Tensorflow 2.4

This repo implemented [LossNet model](https://arxiv.org/pdf/1905.03677.pdf) with TensorFlow 2.4.

* `lossnet_config.py` - A class to help set up config.
* `lossnet_model.py` - The LossNet model architecture.
* `lossnet_prediction_module.py` - The LossNet prediction module class.
* `lossnet_prediction_loss.py` - The LossNet prediction loss class.
* `lossnet_total_loss.py` - The LossNet total loss class.
* `mnist_lossnet_training_e2e_example.py` - A example script runs a simple 2-layer convolution model with MNIST dataset and LossNet model. **This example includes Horovod, AMP and XLA features**.