# data_loader.py
import tensorflow as tf
import numpy as np
import cv2
import config

def data_aug():
    pass

def _get_best_index(anchors, box_wh):
    best_giou = 0
    best_index = 0
    for i in range(len(anchors)):
        min_wh = np.minimum(box_wh, anchors[i])
        max_wh = np.maximum(box_wh, anchors[i])
        giou = (min_wh[0] * min_wh[1]) / (max_wh[0] * max_wh[1])
        if giou > best_giou:
            best_giou = giou
            best_index = i

    return best_index

def _label_init():
    label_width1, label_height1 = config.width // config.stride1, config.height // config.stride1
    label_width2, label_height2 = config.width // config.stride2, config.height // config.stride2
    label_width3, label_height3 = config.width // config.stride3, config.height // config.stride3
    label_1 = np.zeros(
            (label_height1, label_width1, config.anchor_num, 5 + config.class_num), 
            np.float32)
    label_2 = np.zeros(
            (label_height2, label_width2, config.anchor_num, 5 + config.class_num), 
            np.float32)
    label_3 = np.zeros(
            (label_height3, label_width3, config.anchor_num, 5 + config.class_num), 
            np.float32)

    return [label_1, label_2, label_3]

def _get_label(data):
    data = data.astype(np.float)
    labels = _label_init()
    stride_sizes = [config.stride1, config.stride2, config.stride3]

    data = data[2:].reshape(-1, 5)

    for c, cx, cy, box_w, box_h in data:
        best_i = _get_best_index(config.anchors, [box_w, box_h])

        x = int(cx * config.width // stride_sizes[best_i // 3])
        y = int(cy * config.height // stride_sizes[best_i // 3])

        labels[best_i // 3][y, x, best_i % 3, 0:5] = [1, cx, cy, box_w, box_h]
        labels[best_i // 3][y, x, best_i % 3, 5 + int(c) - 1] = 1

    return labels

def _parse_data(line):
    line_split = tf.strings.split([line])
    line_split = tf.sparse.to_dense(line_split, '')
    content = tf.read_file(line_split[0][0])
    image = tf.image.decode_jpeg(content, 3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (config.height, config.width))
    image = image * (1 / 255)
    label1, label2, label3 = tf.py_func(_get_label, [line_split[0][1:]], [tf.float32, tf.float32, tf.float32])

    return image, label1, label2, label3


def load_data(filename, n_repeat=10, buffer_size=50, batch_size=3):
    dataset = tf.data.TextLineDataset([filename])
    dataset = dataset.repeat(n_repeat)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(_parse_data)

    dataset = dataset.batch(batch_size)
    dataset = dataset.make_one_shot_iterator().get_next()

    return dataset

def label_test(image, l1, l2, l3):
    labels = [l1, l2, l3]
    for i, label in enumerate(labels):
        for grid_h in range(len(label)):
            for grid_w in range(len(label[0])):
                for anchor_i in range(len(label[0][0])):
                    if labels[i][grid_h][grid_w][anchor_i][0] == 0:
                        continue
                    x = int(label[grid_h][grid_w][anchor_i][1] * config.width)
                    y = int(label[grid_h][grid_w][anchor_i][2] * config.height)
                    w = int(label[grid_h][grid_w][anchor_i][3] * config.width)
                    h = int(label[grid_h][grid_w][anchor_i][4] * config.height)
                    classes = labels[i][grid_h][grid_w][anchor_i][5:]
                    class_i = list(classes).index(1)
                    xmin = int(x - 1/2 * w)
                    xmax = int(x + 1/2 * w)
                    ymin = int(y - 1/2 * h)
                    ymax = int(y + 1/2 * h)
                    cv2.putText(image, str(config.index2class[class_i+1]), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    return image
                

if __name__ == '__main__':
    dataset = load_data('./data/train.txt')
    
    with tf.Session() as sess:
        images, label1, label2, label3 = sess.run(dataset)

    for i, (image, l1, l2, l3) in enumerate(zip(images, label1, label2, label3)):
        image = label_test(image, l1, l2, l3)
        cv2.imwrite('./data/test%d.jpg' % i, np.clip(image[:,:,::-1] * 255, 0, 255).astype(np.uint8))