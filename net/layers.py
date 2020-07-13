import tensorflow as tf
import cfgs
from net.mobilenet_v3 import _conv_bn_relu, _dwise_conv, _batch_normalization_layer, relu6, hard_swish
import tensorflow.contrib.slim as slim


def mish(x, name='mish'):
    with tf.variable_scope(name):
        mish = x * tf.tanh(tf.math.log(1 + tf.exp(x)))
    return mish


def _bn(inputs, is_training):
    bn = tf.layers.batch_normalization(
        inputs=inputs,
        training=is_training,
        momentum=cfgs.BN_MOMENTUM
    )
    return bn


def leaky_relu(x, name='leaky_relu'):
    return tf.nn.leaky_relu(x, alpha=0.1, name=name)


def _conv(inputs, filters, kernel_size, strides=1, padding='same', activation=mish, is_training=False,
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


def route_group(input_layer, groups, group_id):
    conv_group = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return conv_group[group_id]


def upsampling(inputs, method="deconv", output_dim=24, out_size=None):
    assert method in ['resize', 'deconv', 'complex']

    if method == 'resize':
        input_shape = tf.shape(inputs)
        if out_size is not None:
            output = tf.image.resize_nearest_neighbor(inputs, (out_size[1], out_size[2]))
        else:
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

        up = _upsample_resize(up, output_dim // 2, k_size=3)
        down = _upsample_group_deconv(down, output_dim // 2, group=4)
        output = tf.concat([up, down], axis=3)
        output = _shuffle(output)

    return output


def _dwise_bn_act(inputs, is_training, name):
    out = _dwise_conv(inputs, name=name)
    out = _batch_normalization_layer(out, is_training=is_training, name='{}_bn'.format(name))
    out = hard_swish(out)
    return out


def context_module_dwise(inputs, is_training):
    context_up = _dwise_bn_act(inputs, is_training, 'context_up')
    context_down = _dwise_bn_act(inputs, is_training, 'context_down1')
    context_down = _dwise_bn_act(context_down, is_training, 'context_down2')
    return context_up, context_down


# def context_module_dwise(inputs, is_training):
#     context_up = _dwise_bn_act(inputs, is_training, 'context_up')
#     context_down = _dwise_bn_act(context_up, is_training, 'context_down')
#     return context_up, context_down


# depth_wise
# def detect_module_dwise(inputs, channel, kernel_size, is_training):
#     # up = _conv_bn_relu(inputs, channel, kernel_size, 'detect_up', use_bias=False, is_training=is_training,
#     #                    activation=hard_swish)
#     up = _dwise_bn_act(inputs, is_training, 'detect_up')
#     # rout = route_group(up, 2, 1)
#     context_up, context_down = context_module_dwise(up, is_training)
#     out = tf.concat([up, context_up, context_down], axis=-1)
#     return out


def detect_module_dwise(inputs, channel, kernel_size, is_training):
    out = _dwise_conv(inputs, name='detect_up')
    out = _batch_normalization_layer(out, is_training=is_training, name='detect_up_bn')
    out = leaky_relu(out)
    return out


def context_module_conv(inputs, is_training):
    channel = inputs.shape[-1]
    context_up = _conv(inputs, channel, 3, activation=hard_swish, is_training=is_training)
    context_down = _conv(inputs, channel, 3, activation=hard_swish, is_training=is_training)
    context_down = _conv(context_down, channel, 3, activation=hard_swish, is_training=is_training)
    return context_up, context_down


# conv
def detect_module_conv(inputs, channel, kernel_size, is_training):
    up = _conv_bn_relu(inputs, channel, kernel_size, 'detect_up', use_bias=False, is_training=is_training,
                       activation=hard_swish)
    context_up, context_down = context_module_conv(up, is_training)
    out = tf.concat([up, context_up, context_down], axis=-1)
    return out


def BFP(in_channels, is_training):
    c2, c3, c4, c5 = in_channels
    channel = 24
    p5 = _conv(c5, channel, [1, 1], is_training=is_training)

    up_p5 = upsampling(p5, method='resize')
    f_p5 = up_p5
    reduce_dim_c4 = _conv(c4, channel, [1, 1], is_training=is_training)
    f_p4 = reduce_dim_c4
    p4 = up_p5 + reduce_dim_c4

    up_p4 = upsampling(p4, method='resize')
    reduce_dim_c3 = _conv(c3, channel, [1, 1], is_training=is_training)
    f_p3 = tf.layers.max_pooling2d(reduce_dim_c3, 2, 2, padding='SAME')
    p3 = up_p4 + reduce_dim_c3

    up_p3 = upsampling(p3, method='resize')
    reduce_dim_c2 = _conv(c2, channel, [1, 1], is_training=is_training)
    f_p2 = tf.layers.max_pooling2d(reduce_dim_c2, 4, 4, padding='SAME')
    p2 = 0.25 * (up_p3 + reduce_dim_c2)

    bsf = 0.25 * (f_p2 + f_p3 + f_p4 + f_p5)
    bsf = non_local(bsf, is_training)
    bsf = upsampling(bsf, method='resize', out_size=tf.shape(p2))

    p2 = p2 + bsf
    return p2


def non_local(inputs, is_training):
    batchsize, height, width, channels = inputs.get_shape().as_list()
    g = _conv(inputs, channels, [1, 1], 1, is_training=is_training)
    phi = _conv(inputs, channels, [1, 1], 1, is_training=is_training)
    theta = _conv(inputs, channels, [1, 1], 1, is_training=is_training)

    g_x = tf.reshape(g, [batchsize, channels, -1])
    g_x = tf.transpose(g_x, [0, 2, 1])

    theta_x = tf.reshape(theta, [batchsize, channels, -1])
    theta_x = tf.transpose(theta_x, [0, 2, 1])
    phi_x = tf.reshape(phi, [batchsize, channels, -1])

    f = tf.matmul(theta_x, phi_x)
    f_softmax = tf.nn.softmax(f, -1)
    y = tf.matmul(f_softmax, g_x)
    y = tf.reshape(y, [batchsize, height, width, channels])
    return y
