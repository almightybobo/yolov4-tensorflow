# data_loader.py
import tensorflow as tf
import numpy as np
import cv2
import config

def data_aug():
    pass

def _get_label(data):
    # TODO: 3 labels, only one now
    data = data.astype(np.float)
    label_width, label_height = config.width // config.stride_w, config.height // config.stride_h
    label_1 = np.zeros(
            (label_height, label_width, config.anchor_num, 5 + config.class_num), 
            np.float32)

    width, height = int(data[0]), int(data[1])
    data = data[2:].reshape(-1, 5)
    for i, (c, xmin, xmax, ymin, ymax) in enumerate(data):
        classes = [0] * config.class_num
        classes[int(c)-1] = 1 
        xmin = xmin / width * label_width
        xmax = xmax / width * label_width
        ymin = ymin / height * label_height
        ymax = ymax / height * label_height
        center_x = int((xmin + xmax) // 2)
        center_y = int((ymin + ymax) // 2)
        if i >= config.anchor_num:
            break
        label_1[center_y, center_x, i, :] = [1, ymin, ymax, xmin, xmax] + classes

    return label_1

def _parse_data(line):
    line_split = tf.strings.split([line])
    line_split = tf.sparse.to_dense(line_split, '')
    content = tf.read_file(line_split[0][0])
    image = tf.image.decode_jpeg(content, 3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (config.height, config.width))
    image = image * (1 / 255)
    label = tf.py_func(_get_label, [line_split[0][1:]], tf.float32)
    return image, label


def load_data(filename, n_repeat=10, buffer_size=50, batch_size=3):
    dataset = tf.data.TextLineDataset([filename])
    dataset = dataset.repeat(n_repeat)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(_parse_data)

    dataset = dataset.batch(batch_size)
    dataset = dataset.make_one_shot_iterator().get_next()

    return dataset

def label_test(image, label):
    for grid_h in range(len(label)):
        for grid_w in range(len(label[0])):
            for anchor_i in range(len(label[0][0])):
                if label[grid_h][grid_w][anchor_i][0] == 0:
                    continue
                ymin = int(label[grid_h][grid_w][anchor_i][1] * config.stride_h)
                ymax = int(label[grid_h][grid_w][anchor_i][2] * config.stride_h)
                xmin = int(label[grid_h][grid_w][anchor_i][3] * config.stride_w)
                xmax = int(label[grid_h][grid_w][anchor_i][4] * config.stride_w)
                classes = label[grid_h][grid_w][anchor_i][5:]
                class_i = list(classes).index(1)
                cv2.putText(image, str(config.index2class[class_i+1]), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    return image
                

if __name__ == '__main__':
    dataset = load_data('./data/train.txt')
    
    with tf.Session() as sess:
        batch_images, labels = sess.run(dataset)

    for i, (image, label) in enumerate(zip(batch_images, labels)):
        image = label_test(image, label)
        cv2.imwrite('./data/test%d.jpg' % i, np.clip(image[:,:,::-1] * 255, 0, 255).astype(np.uint8))