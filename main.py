import tensorflow as tf
import os
import config
import data_preprocess
import data_loader
import model
import loss_function

def _get_placeholder(dataset):
    x = tf.placeholder_with_default(dataset[0], shape=[None, config.height, config.width, 3])
    y1 = tf.placeholder_with_default(
            dataset[1],
            shape=[None, config.height // config.stride1, config.width // config.stride1, config.anchor_num, 5+config.class_num])
    y2 = tf.placeholder_with_default(
            dataset[2],
            shape=[None, config.height // config.stride2, config.width // config.stride2, config.anchor_num, 5+config.class_num])
    y3 = tf.placeholder_with_default(
            dataset[3],
            shape=[None, config.height // config.stride3, config.width // config.stride3, config.anchor_num, 5+config.class_num])

    return x, y1, y2, y3

def train_model(train_path, model_path):
    train_dataset = data_loader.load_data(train_path, train=True)
    valid_dataset = data_loader.load_data(train_path, train=False)

    train_x, train_y1, train_y2, train_y3 = _get_placeholder(train_dataset)
    valid_x, valid_y1, valid_y2, valid_y3 = _get_placeholder(valid_dataset)

    with tf.variable_scope('train'):
        train_pred = model.yolov4(train_x)
    with tf.variable_scope('train', reuse=True):
        valid_pred = model.yolov4(valid_x)

    tr_loss = loss_function.yolov4_loss(
            predict_list=train_pred, 
            label_list=[train_y1, train_y2, train_y3])
    val_loss = loss_function.yolov4_loss(
            predict_list=valid_pred, 
            label_list=[valid_y1, valid_y2, valid_y3])

    opt = tf.train.AdamOptimizer().minimize(tr_loss)

    saver = tf.train.Saver(max_to_keep=5)

    best_val_loss, best_val_epoch = None, None

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(config.epochs):
            _, tr_loss_value, val_loss_value = sess.run(
                    [opt, tr_loss, val_loss]) 

            if best_val_loss is None or best_val_loss > val_loss_value:
                best_val_loss = val_loss_value
                best_val_epoch = epoch 
                saver.save(sess, model_path)

            if epoch > best_val_epoch + config.max_stagnation:
                break
            
            print('Epoch: {}'.format(epoch))
            print('Train Loss: {}'.format(tr_loss_value))
            print('Valid Loss: {}'.format(val_loss_value))
        
'''
def inference(test_path):
    test_label, test_feature = arrange_data(test_data) # last 30 data sma30 cannot calculate  
    
    input_x = sess.graph.get_tensor_by_name('Placeholder:0')
    input_y = sess.graph.get_tensor_by_name('Placeholder_1:0')
    output = sess.graph.get_tensor_by_name('train/output/BiasAdd:0')

    test_loss = tf.losses.mean_squared_error(
            labels=tf.expand_dims(input_y, 1), predictions=output)
    test_loss_value = sess.run([test_loss], feed_dict={input_x: test_feature, input_y: test_label}) 
    print(test_loss_value)
'''

if __name__ == '__main__':
    if not os.path.exists(config.train_path):
        data_preprocess.datainfo2txt('./data/VOCtrainval/Annotations/*', './data/VOCtrainval/JPEGImages/', './data/train.txt')
    if not os.path.exists(config.test_path):
        data_preprocess.datainfo2txt('./data/VOCtest/Annotations/*', './data/VOCtest/JPEGImages/', './data/test.txt')
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    train_model(config.train_path, config.model_path)
    