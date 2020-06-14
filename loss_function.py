# loss function
import tensorflow as tf
import numpy as np
import config

'''
def _ciou_loss(predict, label):
    true_ymin, true_ymax = label[..., 1], label[..., 2]
    true_xmin, true_xmax = label[..., 3], label[..., 4]
    pred_ymin, pred_ymax = predict[..., 1], predict[..., 2]
    pred_xmin, pred_xmax = predict[..., 3], predict[..., 3]

    true_left_top = true_xy - true_wh / 2
    true_right_bottom = true_xy + true_wh / 2
    pred_left_top = pred_xy - true_wh / 2
    pred_right_bottom = pred_xy + true_wh / 2
'''

def _get_iou(xy, wh, label):
    pred_xy = tf.expand_dims(xy, -2)
    pred_wh = tf.expand_dims(wh, -2)

    true_xy = label[..., 0:2]
    true_wh = label[..., 2:4]

    pred_left_top = pred_xy - pred_wh / 2
    true_left_top = true_xy - true_wh / 2
    max_left_top = tf.maximum(pred_left_top, true_left_top)

    pred_right_bottom = pred_xy + pred_wh / 2
    true_right_bottom = true_xy + true_wh / 2
    min_right_bottom = tf.minimum(pred_right_bottom, true_right_bottom)

    intersection_wh = tf.maximum(min_right_bottom - max_left_top, 0.0)
    
    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    true_area = tf.expand_dims(true_wh[..., 0] * true_wh[..., 1], axis=0)

    iou = intersection_area / (pred_area + true_area - intersection_area + 1e-10)

    return iou

def _get_low_iou_mask(xy, wh, label, ignore_thresh):
    true_conf = label[..., 4]

    low_iou_mask = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
    batch_size = label.shape[0]
    
    def loop_cond(index, low_iou_mask):
        return tf.less(index, batch_size)        
    def loop_body(index, low_iou_mask):
        label_valid = tf.boolean_mask(
                label[index], 
                tf.cast(true_conf[index, ..., 0], tf.bool))

        iou = _get_iou(xy[index], wh[index], label_valid)

        best_giou = tf.reduce_max(iou, axis=-1)
        low_iou_mask_tmp = best_giou < ignore_thresh
        low_iou_mask_tmp = tf.expand_dims(low_iou_mask_tmp, -1)
        low_iou_mask = low_iou_mask.write(index, low_iou_mask_tmp)
        return index+1, low_iou_mask

    _, low_iou_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, low_iou_mask])
    low_iou_mask = low_iou_mask.stack()
    return low_iou_mask

def _get_low_prob_mask(prob, prob_thresh):
    max_prob = tf.reduce_max(prob, axis=-1, keepdims=True)
    low_prob_mask = max_prob < prob_thresh        
    return low_prob_mask

def _get_low_iou_prob_mask(xy, wh, prob, label, ignore_thresh, prob_thresh):
    low_iou_mask = _get_low_iou_mask(xy, wh, label, ignore_thresh)
    low_prob_mask = _get_low_prob_mask(prob, prob_thresh)        
    low_iou_prob_mask = tf.cast(tf.math.logical_or(low_iou_mask, low_prob_mask), tf.float32)
    return low_iou_prob_mask

def _get_ciou_loss():
    pass

def _get_yolov4_loss(pred_xy, pred_wh, pred_conf, pred_prob, label):
    batch_size = pred_xy.shape[0]

    area = pred_wh[..., 0] * pred_wh[..., 1]
    area = tf.where(tf.math.greater(area, 0),
            area, tf.math.square(area))

    low_iou_prob_mask = _get_low_iou_prob_mask(
            pred_xy, pred_wh, pred_prob, label, config.ignore_thresh, config.prob_thresh)
    no_obj_mask = 1.0 - label[..., 4]
    no_obj_conf_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label[:,:,:,:,4], 
            logits=pred_conf) * area * no_obj_mask * low_iou_prob_mask

    obj_mask = label[..., 4]        
    obj_conf_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label[:,:,:,:,4], 
            logits=pred_conf) * obj_mask        
    
    conf_loss = no_obj_conf_loss + obj_conf_loss
    conf_loss = tf.clip_by_value(conf_loss, 0.0, 1e3)
    conf_loss = tf.reduce_sum(conf_loss) / batch_size

    '''
    # ciou_loss
    yi_true_ciou = tf.where(tf.math.less(yi_true[..., 0:4], 1e-10),
                                            tf.ones_like(yi_true[..., 0:4]), yi_true[..., 0:4])
    pre_xy = tf.where(tf.math.less(xy, 1e-10),
                                            tf.ones_like(xy), xy)
    pre_wh = tf.where(tf.math.less(wh, 1e-10),
                                            tf.ones_like(wh), wh)
    ciou_loss = self.__my_CIOU_loss(pre_xy, pre_wh, yi_true_ciou)
    ciou_loss = tf.where(tf.math.greater(obj_mask, 0.5), ciou_loss, tf.zeros_like(ciou_loss))
    ciou_loss = tf.square(ciou_loss * obj_mask) * iou_normalizer
    ciou_loss = tf.clip_by_value(ciou_loss, 0, 1e3)
    ciou_loss = tf.reduce_sum(ciou_loss) / N
    ciou_loss = tf.clip_by_value(ciou_loss, 0, 1e4)
    conf_loss = no_obj_conf_loss + obj_conf_loss
    conf_loss = tf.clip_by_value(conf_loss, 0.0, 1e3)
    conf_loss = tf.reduce_sum(conf_loss) / batch_size
    '''
    xy_loss = obj_mask * tf.square(label[..., 0:2] - pred_xy) 
    xy_loss = tf.reduce_sum(xy_loss) / batch_size
    xy_loss = tf.clip_by_value(xy_loss, 0.0, 1e4)

    refine_true_wh = tf.where(
            tf.math.less(label[..., 2:4], 1e-10),
            tf.ones_like(label[..., 2:4]), 
            label[..., 2:4])
    refine_pred_wh = tf.where(
            tf.math.less(pred_wh, 1e-10),
            tf.ones_like(pred_wh),
            pred_wh)

    refine_true_wh = tf.math.log(tf.clip_by_value(refine_true_wh, 1e-10, 1e10))
    refine_pred_wh = tf.math.log(tf.clip_by_value(refine_pred_wh, 1e-10, 1e10))

    wh_loss = obj_mask * tf.square(refine_true_wh - refine_pred_wh)
    wh_loss = tf.reduce_sum(wh_loss) / batch_size
    wh_loss = tf.clip_by_value(wh_loss, 0.0, 1e4)

    prob_score = pred_prob * pred_conf
    high_score_mask = tf.cast(prob_score > config.score_thresh, tf.float32)
    
    no_obj_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label[..., 5:5+config.class_num],
            logits=pred_prob) * low_iou_prob_mask * no_obj_mask * high_score_mask
    
    obj_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label[..., 5:5+config.class_num],
            logits=pred_prob) * obj_mask

    class_loss = (no_obj_class_loss + obj_class_loss)
    class_loss = tf.reduce_sum(class_loss) / batch_size    

    loss_total = xy_loss + wh_loss + conf_loss + class_loss + ciou_loss

    return loss_total

def _decode_predict(predict, anchors):
    shape = predict.shape
    predict = tf.reshape(
            predict, 
            [shape[0], shape[1], shape[2], config.anchor_num, 5+config.class_num])
    xy, wh, conf, prob = tf.split(predict, [2, 2, 1, config.class_num], axis=-1)
    offset_x = tf.range(shape[2], dtype=tf.float32)
    offset_y = tf.range(shape[1], dtype=tf.float32)
    offset_x, offset_y = tf.meshgrid(offset_x, offset_y)
    offset_x = tf.reshape(offset_x, (-1, 1))
    offset_y = tf.reshape(offset_y, (-1, 1))
    offset_xy = tf.concat([offset_x, offset_y], axis=-1)
    offset_xy = tf.reshape(offset_xy, [shape[1], shape[2], 1, 2])

    xy = tf.math.sigmoid(xy) + offset_xy    
    xy = xy / [shape[2], shape[1]]

    wh = tf.math.exp(wh) * anchors
    wh = wh / [config.width, config.height]

    return xy, wh, conf, prob

def yolov4_loss(predict_list, label_list):
    total_loss = 0
    anchors = [config.anchors[0::3], config.anchors[1::3], config.anchors[2::3]]
    for i, (predict, label) in enumerate(zip(predict_list, label_list)):
        xy, wh, conf, prob = _decode_predict(predict, anchors[i])
        total_loss += _get_yolov4_loss(xy, wh, conf, prob, label)

    return total_loss


    


if __name__ == '__main__':
    print('loss_function')
