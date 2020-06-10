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

    for _ in range(8):
        net = module.yolo_block(net, 256)
    net = module.conv(net, 1024, 3, 2)

    for _ in range(4):
        net = module.yolo_block(net, 512)

    net = module.yolo_block(net, 512)
    net = module.conv(net, 512, 1, 1)

    # SPP
    pool_5 = tf.nn.max_pool(inputs, [1, 5, 5, 1], [1, 1, 1, 1], 'SAME')
    pool_9 = tf.nn.max_pool(inputs, [1, 9, 9, 1], [1, 1, 1, 1], 'SAME')
    pool_13 = tf.nn.max_pool(inputs, [1, 13, 13, 1], [1, 1, 1, 1], 'SAME')
    net = tf.concat([pool_13, pool_9, pool_5, net], -1)

    net = module.conv(net, 512, 1, 1)
    net = module.conv(net, 1024, 3, 1)
    net = module.conv(net, 512, 1, 1)
    
    # TODO: 3 output, only 1 now

    return net