# model.py
import tensorflow as tf
import module
import config

def yolov4(input_img):
    net = module.conv(input_img, 32, 3, 1)

    net = module.conv(net, 64, 3, 2)
    route = module.conv(net, 64, 1, 1)
    net = module.conv(net, 64, 1, 1)
    tmp = net
    net = module.conv(net, 32, 1, 1)
    net = module.conv(net, 64, 3, 1)
    net = tmp + net
    net = module.conv(net, 64, 1, 1)
    net = tf.concat([net, route], -1) 
    net = module.conv(net, 64, 1, 1)

    net = module.conv(net, 128, 3, 2)
    route = module.conv(net, 64, 1, 1)
    net = module.conv(net, 64, 1, 1)
    for _ in range(2):
        tmp = net
        net = module.conv(net, 64, 1, 1)
        net = module.conv(net, 64, 3, 1)
        net = tmp + net
    net = module.conv(net, 64, 1, 1)
    net = tf.concat([net, route], -1) 
    net = module.conv(net, 128, 1, 1)
    
    net = module.conv(net, 256, 3, 2)
    route = module.conv(net, 128, 1, 1)
    net = module.conv(net, 128, 1, 1)
    for _ in range(8):
        tmp = net
        net = module.conv(net, 128, 1, 1)
        net = module.conv(net, 128, 3, 1)
        net = tmp + net
    net = module.conv(net, 128, 1, 1)
    net = tf.concat([net, route], -1) 
    net = module.conv(net, 256, 1, 1)

    for_output1 = net

    net = module.conv(net, 512, 3, 2)
    route = module.conv(net, 256, 1, 1)
    net = module.conv(net, 256, 1, 1)
    for _ in range(8):
        tmp = net
        net = module.conv(net, 256, 1, 1)
        net = module.conv(net, 256, 3, 1)
        net = tmp + net
    net = module.conv(net, 256, 1, 1)
    net = tf.concat([net, route], -1) 
    net = module.conv(net, 512, 1, 1)

    for_output2 = net

    net = module.conv(net, 1024, 3, 2)
    route = module.conv(net, 512, 1, 1)
    net = module.conv(net, 512, 1, 1)
    for _ in range(4):
        tmp = net
        net = module.conv(net, 512, 1, 1)
        net = module.conv(net, 512, 3, 1)
        net = tmp + net
    net = module.conv(net, 512, 1, 1)
    net = tf.concat([net, route], -1) 
    net = module.conv(net, 1024, 1, 1)

    # SPP
    pool_5 = tf.nn.max_pool(net, [1, 5, 5, 1], [1, 1, 1, 1], 'SAME')
    pool_9 = tf.nn.max_pool(net, [1, 9, 9, 1], [1, 1, 1, 1], 'SAME')
    pool_13 = tf.nn.max_pool(net, [1, 13, 13, 1], [1, 1, 1, 1], 'SAME')
    net = tf.concat([pool_13, pool_9, pool_5, net], -1)

    net = module.conv(net, 512, 1, 1)
    net = module.conv(net, 1024, 3, 1)
    net = module.conv(net, 512, 1, 1)
    output3 = net
    
    net = module.conv(net, 256, 1, 1)
    shape = tf.shape(net)
    out_height, out_width = shape[1]*2, shape[2]*2
    net = tf.compat.v1.image.resize_nearest_neighbor(net, (out_height, out_width))
    for_output2 = module.conv(for_output2, 256, 1, 1)
    net = tf.concat([for_output2, net], -1)

    net = module.conv(net, 256, 1, 1)
    net = module.conv(net, 512, 3, 1)
    net = module.conv(net, 256, 1, 1)
    net = module.conv(net, 512, 3, 1)
    net = module.conv(net, 256, 1, 1)
    output2 = net

    net = module.conv(net, 128, 1, 1)
    shape = tf.shape(net)
    out_height, out_width = shape[1]*2, shape[2]*2
    net = tf.compat.v1.image.resize_nearest_neighbor(net, (out_height, out_width))
    for_output1 = module.conv(for_output1, 128, 1, 1)
    net = tf.concat([for_output1, net], -1)

    net = module.conv(net, 128, 1, 1)
    net = module.conv(net, 256, 3, 1)
    net = module.conv(net, 128, 1, 1)
    net = module.conv(net, 256, 3, 1)
    net = module.conv(net, 128, 1, 1)
    output1 = net

    net = module.conv(output1, 256, 3, 1)
    net = module.conv(net, 3*(4+1+config.class_num), 1, 1,
            normalizer_fn=None, activation_fn=None, 
            biases_initializer=tf.zeros_initializer())
    label3 = net

    net = module.conv(output1, 256, 3, 2)
    net = tf.concat([net, output2], -1)
    net = module.conv(net, 256, 1, 1)
    net = module.conv(net, 512, 3, 1)
    net = module.conv(net, 256, 1, 1)
    net = module.conv(net, 512, 3, 1)
    net = module.conv(net, 256, 1, 1)
    tmp_route = net

    net = module.conv(net, 512, 3, 1)
    net = module.conv(net, 3*(4+1+config.class_num), 1, 1,
            normalizer_fn=None, activation_fn=None, 
            biases_initializer=tf.zeros_initializer())
    label2 = net

    net = module.conv(tmp_route, 512, 3, 2)
    net = tf.concat([net, output3], -1)
    net = module.conv(net, 512, 1, 1)
    net = module.conv(net, 1024, 3, 1)
    net = module.conv(net, 512, 1, 1)
    net = module.conv(net, 1024, 3, 1)
    net = module.conv(net, 512, 1, 1)
    net = module.conv(net, 1024, 3, 1)
    net = module.conv(net, 3*(4+1+config.class_num), 1, 1,
            normalizer_fn=None, activation_fn=None, 
            biases_initializer=tf.zeros_initializer())
    label1 = net
    
    return label1, label2, label3

if __name__ == '__main__':
    a = tf.placeholder(tf.float32, shape=[None, 608, 608, 3])
    yolov4(a)