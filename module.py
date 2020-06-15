import tensorflow as tf
from tensorflow.contrib import slim
import config

def yolo_block(inputs, filter_size):
    route = inputs
    net = conv(inputs, filter_size, 1, 1)
    net = conv(net, filter_size*2, 3, 1)
    net = tf.concat([net, route], -1)
    return net

def conv(
        inputs, 
        output_channels, 
        kernel_size, 
        stride, 
        padding='SAME', 
        normalizer_fn=slim.batch_norm,
        normalizer_params=config.batch_norm_params,
        biases_initializer=None,
        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
        weights_regularizer=slim.l2_regularizer(config.weight_decay)):
    inputs = slim.conv2d(
            inputs, 
            output_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            biases_initializer=biases_initializer,
            activation_fn=activation_fn,
            weights_regularizer=weights_regularizer)  
    return inputs
