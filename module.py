import tensorflow as tf
from tensorflow.contrib import slim

def yolo_block(inputs, filter_size):
    route = inputs
    net = conv(inputs, filter_size, 1, 1)
    net = conv(net, filter_size*2, 3, 1)
    net = tf.concat([net, route], -1)
    return net

def conv(inputs, output_channels, kernel_size, stride):
    inputs = slim.conv2d(
            inputs, 
            output_channels, 
            kernel_size, 
            stride=stride, 
            padding='SAME')  
    return inputs
