# model.py
import tensorflow as tf
import module

def yolov4(input_img):
    net = module.conv(input_img, 32, 3, 1)
    net = module.conv(net, 64, 3, 2)
    net = module.yolo_block(net, 32)
    net = module.conv(net, 128, 3, 2)
    net = module.yolo_block(net, 64)
    net = module.yolo_block(net, 64)
    net = module.conv(net, 256, 3, 2)

    for _ in range(8):
        net = module.yolo_block(net, 128)
    net = module.conv(net, 512, 3, 2)
    for_output1 = net

    for _ in range(8):
        net = module.yolo_block(net, 256)
    net = module.conv(net, 1024, 3, 2)
    for_output2 = net

    for _ in range(4):
        net = module.yolo_block(net, 512)

    net = module.yolo_block(net, 512)
    net = module.conv(net, 512, 1, 1)

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

    return output1, output2, output3