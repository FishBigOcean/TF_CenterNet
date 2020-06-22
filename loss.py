import tensorflow as tf
import cfgs


# # 目标的高斯分布，表示目标的中心点
# batch_hm = np.zeros(
#     (cfgs.BATCH_SIZE, cfgs.INPUT_IMAGE_H // cfgs.DOWN_RATIO, cfgs.INPUT_IMAGE_W // cfgs.DOWN_RATIO, cfgs.NUM_CLASS),
#     dtype=np.float32)
# # 目标的高度和宽度
# batch_wh = np.zeros((cfgs.BATCH_SIZE, cfgs.MAX_OBJ, 2), dtype=np.float32)
# # 目标中心整数化时的量化误差
# batch_reg = np.zeros((cfgs.BATCH_SIZE, cfgs.MAX_OBJ, 2), dtype=np.float32)
# # 1有目标 0没有目标
# batch_reg_mask = np.zeros((cfgs.BATCH_SIZE, cfgs.MAX_OBJ), dtype=np.float32)
# # 目标关键点在2D heatmap中对应的1D heatmap的索引
# batch_ind = np.zeros((cfgs.BATCH_SIZE, cfgs.MAX_OBJ), dtype=np.float32)

def focal_loss(hm_pred, hm_true):
    pos_mask = tf.cast(tf.equal(hm_true, 1.), dtype=tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1.), dtype=tf.float32)
    neg_weights = tf.pow(1. - hm_true, 4)

    pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-5, 1. - 1e-5)) * tf.pow(1. - hm_pred, 2) * pos_mask
    neg_loss = -tf.log(tf.clip_by_value(1. - hm_pred, 1e-5, 1. - 1e-5)) * tf.pow(hm_pred, 2.0) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss) * cfgs.HM_POS_WEIGHT
    neg_loss = tf.reduce_sum(neg_loss) * cfgs.HM_NEG_WEIGHT

    loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return loss


def reg_l1_loss(y_pred, y_true, indices, mask):
    b = tf.shape(y_pred)[0]
    k = tf.shape(indices)[1]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.batch_gather(y_pred, indices)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    loss = total_loss * 2 / (tf.reduce_sum(mask) + 1e-5)
    return loss


def smooth_l1_loss(y_pred, y_true, indices, mask, sigma=cfgs.SIGMA):
    b = tf.shape(y_pred)[0]
    k = tf.shape(indices)[1]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.batch_gather(y_pred, indices)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    sigma_2 = sigma ** 2
    diff = y_pred * mask - y_true * mask
    abs_diff = tf.abs(diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. / sigma_2)))
    loss_box = tf.pow(diff, 2) * (sigma_2 / 2.0) * smoothL1_sign + (abs_diff - (0.5 / sigma_2)) * (
            1.0 - smoothL1_sign)
    total_loss = tf.reduce_sum(loss_box)
    loss = total_loss * 2 / (tf.reduce_sum(mask) + 1e-5)
    return loss


def cross_entropy_loss(y_pred, y_true, mask):
    y_true = tf.cast(tf.batch_gather(y_true, tf.zeros([cfgs.BATCH_SIZE, 1], dtype=tf.int32)), dtype=tf.int32)
    cls_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    return cls_loss


def giou(self, boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / tf.maximum(union_area, 1e-5)

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / tf.maximum(enclose_area, 1e-5)

    return giou
