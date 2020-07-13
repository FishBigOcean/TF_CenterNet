import tensorflow as tf
import cfgs
import loss
from net import resnet, mobilenet, mobilenet_v3, cspnet
from net.layers import _conv, upsampling, detect_module_dwise, _shuffle, detect_module_conv, BFP
import numpy as np


class CenterNet():
    def __init__(self, inputs, is_training):
        self.is_training = is_training
        try:
            self.pred_hm, self.pred_wh, self.pred_reg = self._build_model(inputs)
            self.pred_wh = tf.exp(self.pred_wh)
        except:
            raise NotImplementedError("Can not build up centernet network!")

    def _build_model(self, inputs):
        with tf.variable_scope('mobilenet'):
            # mobilenet v2
            # c2, c3, c4, c5 = mobilenet.MobileNetV2(inputs=inputs, is_training=self.is_training).forward()

            # mobilenet v3
            # c2, c3, c4, c5 = mobilenet_v3.mobilenet_v3_small(inputs=inputs, is_training=self.is_training)

            # cspdarknet_dw
            # c2, c3, c4, c5 = cspnet.cspdarknet53_tiny_dwise(inputs=inputs, is_training=self.is_training)

            # cspdarknet_dw_focus
            # c2, c3, c4, c5 = cspnet.cspdarknet53_tiny_dwise_focus(inputs=inputs, is_training=self.is_training)

            # # cspdarknet
            c2, c3, c4, c5 = cspnet.cspdarknet53_tiny(inputs=inputs, is_training=self.is_training)

            channel = 24
            # p5 = _conv(c5, channel, [1, 1], is_training=self.is_training)
            #
            # up_p5 = upsampling(p5, method='resize')
            # reduce_dim_c4 = _conv(c4, channel, [1, 1], is_training=self.is_training)
            # p4 = up_p5 + reduce_dim_c4
            #
            # up_p4 = upsampling(p4, method='resize')
            # reduce_dim_c3 = _conv(c3, channel, [1, 1], is_training=self.is_training)
            # p3 = up_p4 + reduce_dim_c3
            #
            # up_p3 = upsampling(p3, method='resize')
            # reduce_dim_c2 = _conv(c2, channel, [1, 1], is_training=self.is_training)
            # p2 = up_p3 + reduce_dim_c2

            p2 = BFP([c2, c3, c4, c5], self.is_training)


            features = _conv(p2, channel, [3, 3], is_training=self.is_training)

            # features = detect_module_dwise(p2, channel, [3, 3], is_training=self.is_training)

            # features = detect_module_conv(p2, channel, [3, 3], is_training=self.is_training)

        with tf.variable_scope('detector'):
            hm = tf.layers.conv2d(features, cfgs.NUM_CLASS, 1, 1, padding='valid', activation=tf.nn.sigmoid,
                                  bias_initializer=tf.constant_initializer(-np.log(99.)), name='hm')
            # tf.initializers.constant(-2.19)  tf.constant_initializer(-np.log(99.))

            wh = tf.layers.conv2d(features, 2, 1, 1, padding='valid', activation=None, name='wh')

            if cfgs.ADD_REG:
                reg = tf.layers.conv2d(features, 2, 1, 1, padding='valid', activation=None, name='reg')
                return hm, wh, reg

            return hm, wh, None

    def compute_loss(self, true_hm, true_wh, true_reg, reg_mask, ind):
        hm_loss = loss.focal_loss(self.pred_hm, true_hm) * cfgs.HM_LOSS_WEIGHT
        wh_loss = loss.reg_l1_loss(self.pred_wh, true_wh, ind, reg_mask) * cfgs.WH_LOSS_WEIGHT
        if cfgs.ADD_REG:
            reg_loss = loss.reg_l1_loss(self.pred_reg, true_reg, ind, reg_mask) * cfgs.REG_LOSS_WEIGHT
        else:
            reg_loss = tf.constant(0.0)
        return hm_loss, wh_loss, reg_loss
