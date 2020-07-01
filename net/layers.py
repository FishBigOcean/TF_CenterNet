import tensorflow as tf
import cfgs
from net.mobilenet_v3 import _conv_bn_relu, _dwise_conv, _batch_normalization_layer, relu6, hard_swish
import tensorflow.contrib.slim as slim


def _bn(inputs, is_training):
    bn = tf.layers.batch_normalization(
        inputs=inputs,
        training=is_training,
        momentum=cfgs.BN_MOMENTUM
    )
    return bn


def _conv(inputs, filters, kernel_size, strides=1, padding='same', activation=tf.nn.relu6, is_training=False,
          use_bn=True):
    if use_bn:
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer=cfgs.WEIGHT_INITIALIZER,
            kernel_regularizer=cfgs.WEIGHT_REGULARIZER
        )
        conv = _bn(conv, is_training)
    else:
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=cfgs.WEIGHT_INITIALIZER,
            kernel_regularizer=cfgs.WEIGHT_REGULARIZER
        )
    if activation is not None:
        conv = activation(conv)
    return conv


def _upsample_resize(inputs, dim, k_size=3):
    input_shape = tf.shape(inputs)
    upsampled_conv = tf.layers.separable_conv2d(inputs, dim, [k_size, k_size], padding='SAME')
    upsampled_conv = tf.image.resize_nearest_neighbor(upsampled_conv, (input_shape[1] * 2, input_shape[2] * 2))

    return upsampled_conv


def _upsample_group_deconv(inputs, dim, group=4):
    '''
    group devonc

    :param fm: input feature
    :param group:
    :return:
    '''
    sliced_fms = tf.split(inputs, num_or_size_splits=group, axis=3)

    deconv_fms = []
    for i in range(group):
        cur_upsampled_conv = tf.layers.conv2d_transpose(sliced_fms[i], dim // group, [4, 4], strides=2, padding='SAME')
        deconv_fms.append(cur_upsampled_conv)

    deconv_fm = tf.concat(deconv_fms, axis=3)

    return deconv_fm


def _shuffle(inputs, group=2):
    with tf.name_scope('shuffle'):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = shape[3] // group

        inputs = tf.reshape(inputs,
                            [batch_size, height, width, group, depth])  # shape [batch_size, height, width, 2, depth]
        inputs = tf.transpose(inputs, [0, 1, 2, 4, 3])
        inputs = tf.reshape(inputs, [batch_size, height, width, group * depth])

        return inputs


def upsampling(inputs, method="deconv", output_dim=24):
    assert method in ['resize', 'deconv', 'complex']

    if method == 'resize':
        input_shape = tf.shape(inputs)
        output = tf.image.resize_nearest_neighbor(inputs, (input_shape[1] * 2, input_shape[2] * 2))

    if method == 'deconv':
        numm_filter = inputs.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=numm_filter,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=cfgs.WEIGHT_INITIALIZER,
            kernel_regularizer=cfgs.WEIGHT_REGULARIZER
        )

    if method == 'complex':
        input_dim = tf.shape(inputs)[3]
        up = inputs[:, :, :, :input_dim // 2]
        down = inputs[:, :, :, input_dim // 2:]

        up = _upsample_resize(up, output_dim//2, k_size=3)
        down = _upsample_group_deconv(down, output_dim//2, group=4)
        output = tf.concat([up, down], axis=3)
        output = _shuffle(output)

    return output


def _dwise_bn_act(inputs, is_training, name):
    out = _dwise_conv(inputs, name=name)
    out = _batch_normalization_layer(out, is_training=is_training, name='{}_bn'.format(name))
    out = hard_swish(out)
    return out


def context_module(inputs, is_training):
    context_up = _dwise_bn_act(inputs, is_training, 'context_up')
    context_down = _dwise_bn_act(inputs, is_training, 'context_down1')
    context_down = _dwise_bn_act(context_down, is_training, 'context_down2')
    return context_up, context_down


# depth_wise
# def detect_module(inputs, channel, kernel_size, is_training):
#     # up = _conv_bn_relu(inputs, channel, kernel_size, 'detect_up', use_bias=False, is_training=is_training,
#     #                    activation=hard_swish)
#     up = _dwise_bn_act(inputs, is_training, 'detect_up')
#     context_up, context_down = context_module(up, is_training)
#     out = tf.concat([up, context_up, context_down], axis=-1)
#     return out


def context_module_conv(inputs, is_training):
    channel = inputs.shape[-1]
    context_up = _conv(inputs, channel, 3, activation=hard_swish, is_training=is_training)
    context_down = _conv(inputs, channel, 3, activation=hard_swish, is_training=is_training)
    context_down = _conv(context_down, channel, 3, activation=hard_swish, is_training=is_training)
    return context_up, context_down

# conv
def detect_module(inputs, channel, kernel_size, is_training):
    up = _conv_bn_relu(inputs, channel, kernel_size, 'detect_up', use_bias=False, is_training=is_training,
                       activation=hard_swish)
    context_up, context_down = context_module_conv(up, is_training)
    out = tf.concat([up, context_up, context_down], axis=-1)
    return out