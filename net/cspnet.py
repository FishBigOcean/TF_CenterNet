import tensorflow as tf
import cfgs


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _batch_normalization_layer(inputs, momentum=cfgs.BN_MOMENTUM, epsilon=1e-3, is_training=True, name='bn',
                               reuse=None):
    return tf.layers.batch_normalization(inputs=inputs,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         center=True,
                                         training=is_training,
                                         name=name,
                                         reuse=reuse)


def _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=False, strides=1, reuse=None, padding="SAME"):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=cfgs.WEIGHT_INITIALIZER,
        padding=padding,  # ('SAME' if strides == 1 else 'VALID'),
        kernel_regularizer=cfgs.WEIGHT_REGULARIZER, use_bias=use_bias, name=name,
        reuse=reuse)
    return conv


def _conv_1x1_bn(inputs, filters_num, name, use_bias=True, is_training=True, reuse=None):
    kernel_size = 1
    strides = 1
    x = _conv2d_layer(inputs, filters_num, kernel_size, name=name + "/conv", use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, is_training=is_training, name=name + '/bn', reuse=reuse)
    return x


def _conv_bn_relu(inputs, filters_num, kernel_size, name, use_bias=True, strides=1, is_training=True, activation=None,
                  reuse=None):
    x = _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, is_training=is_training, name=name + '/bn', reuse=reuse)
    x = activation(x)
    return x


def _dwise_conv(inputs, k_h=3, k_w=3, depth_multiplier=1, strides=(1, 1),
                padding='SAME', name='dwise_conv', use_bias=False,
                reuse=None):
    kernel_size = (k_w, k_h)
    in_channel = inputs.get_shape().as_list()[-1]
    filters = int(in_channel * depth_multiplier)
    return tf.layers.separable_conv2d(inputs, filters, kernel_size,
                                      strides=strides, padding=padding,
                                      data_format='channels_last', dilation_rate=(1, 1),
                                      depth_multiplier=depth_multiplier, activation=None,
                                      use_bias=use_bias, name=name, reuse=reuse,
                                      depthwise_initializer=cfgs.WEIGHT_INITIALIZER,
                                      pointwise_initializer=cfgs.WEIGHT_INITIALIZER,
                                      depthwise_regularizer=cfgs.WEIGHT_REGULARIZER,
                                      pointwise_regularizer=cfgs.WEIGHT_REGULARIZER
                                      )


def mobilenet_v3_block(input, k_s, expansion_ratio, output_dim, stride, name, is_training=True,
                       use_bias=True, shortcut=True, activatation="RE", ratio=16, se=False,
                       reuse=None):
    bottleneck_dim = expansion_ratio

    with tf.variable_scope(name, reuse=reuse):
        # pw mobilenetV2
        net = _conv_1x1_bn(input, bottleneck_dim, name="pw", use_bias=False, is_training=is_training)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # dw
        net = _dwise_conv(net, k_w=k_s, k_h=k_s, strides=[stride, stride], name='dw',
                          use_bias=use_bias, reuse=reuse)

        net = _batch_normalization_layer(net, is_training=is_training, name='dw_bn', reuse=reuse)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # squeeze and excitation
        if se:
            channel = net.get_shape().as_list()[-1]
            net = _squeeze_excitation_layer(net, out_dim=channel, ratio=ratio, layer_name='se_block')

        # pw & linear
        net = _conv_1x1_bn(net, output_dim, name="pw_linear", use_bias=use_bias, is_training=is_training)

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            net += input
            net = tf.identity(net, name='block_output')

    return net


def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)


def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish


def leaky_relu(x, name='leaky_relu'):
    return tf.nn.leaky_relu(x, alpha=0.1, name=name)


def mish(x, name='mish'):
    with tf.variable_scope(name):
        mish = x * tf.tanh(tf.math.log(1 + tf.exp(x)))
    return mish


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid


def _fully_connected_layer(inputs, units, name="fc", activation=None, use_bias=True, reuse=None):
    return tf.layers.dense(inputs, units, activation=activation, use_bias=use_bias,
                           kernel_initializer=cfgs.WEIGHT_INITIALIZER, name=name,
                           reuse=reuse, kernel_regularizer=cfgs.WEIGHT_REGULARIZER)


def _global_avg(inputs, pool_size, strides, padding='valid', name='global_avg'):
    return tf.layers.average_pooling2d(inputs, pool_size, strides,
                                       padding=padding, data_format='channels_last', name=name)


def _squeeze_excitation_layer(input, out_dim, ratio, layer_name, is_training=True, reuse=None):
    with tf.variable_scope(layer_name, reuse=reuse):
        squeeze = _global_avg(input, pool_size=input.get_shape()[1:-1], strides=1)

        excitation = _fully_connected_layer(squeeze, units=out_dim / ratio, name=layer_name + '_excitation1',
                                            reuse=reuse)
        excitation = relu6(excitation)
        excitation = _fully_connected_layer(excitation, units=out_dim, name=layer_name + '_excitation2', reuse=reuse)
        excitation = hard_sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input * excitation
        return scale


def _dwise_bn_act(inputs, is_training, name, activation=relu6):
    out = _dwise_conv(inputs, name=name)
    out = _batch_normalization_layer(out, is_training=is_training, name='{}_bn'.format(name))
    out = activation(out)
    return out


def route_group(input_layer, groups, group_id):
    conv_group = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return conv_group[group_id]


def focus(inputs, out_channel, is_training):
    # Focus wh information into c-space
    shape = tf.shape(inputs)
    inputs = tf.reshape(inputs, [shape[0], shape[1] // 2, shape[2] // 2, shape[3] * 4])
    outputs = _conv_bn_relu(inputs, filters_num=out_channel, kernel_size=3, name='init',
                           use_bias=False, strides=1, is_training=is_training, activation=leaky_relu)
    return outputs


def cspdarknet53_tiny(inputs, is_training):
    init_conv_out = _make_divisible(32 * cfgs.CSPNET_SHRINK)
    inputs = _conv_bn_relu(inputs, filters_num=init_conv_out, kernel_size=3, name='init',
                           use_bias=False, strides=2, is_training=is_training, activation=hard_swish)

    conv_out = _make_divisible(64 * cfgs.CSPNET_SHRINK)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out, kernel_size=3, name='c2_0',
                           use_bias=False, strides=2, is_training=is_training, activation=relu6)

    conv_out = _make_divisible(32 * cfgs.CSPNET_SHRINK)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out * 2, kernel_size=3, name='c2_1',
                           use_bias=False, strides=1, is_training=is_training, activation=relu6)
    route = inputs
    inputs = route_group(inputs, 2, 1)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out, kernel_size=3, name='c2_up_1',
                           use_bias=False, strides=1, is_training=is_training, activation=relu6)
    route_1 = inputs
    inputs = _conv_bn_relu(inputs, filters_num=conv_out, kernel_size=3, name='c2_up_2',
                           use_bias=False, strides=1, is_training=is_training, activation=relu6)
    inputs = tf.concat([inputs, route_1], axis=-1)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out * 2, kernel_size=3, name='c2_up_3',
                           use_bias=False, strides=1, is_training=is_training, activation=relu6)
    inputs = tf.concat([route, inputs], axis=-1)
    c2 = inputs
    inputs = tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME')

    conv_out = _make_divisible(64 * cfgs.CSPNET_SHRINK)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out * 2, kernel_size=3, name='c3_1',
                           use_bias=False, strides=1, is_training=is_training, activation=relu6)
    route = inputs
    inputs = route_group(inputs, 2, 1)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out, kernel_size=3, name='c3_up_1',
                           use_bias=False, strides=1, is_training=is_training, activation=relu6)
    route_1 = inputs
    inputs = _conv_bn_relu(inputs, filters_num=conv_out, kernel_size=3, name='c3_up_2',
                           use_bias=False, strides=1, is_training=is_training, activation=relu6)
    inputs = tf.concat([inputs, route_1], axis=-1)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out * 2, kernel_size=3, name='c3_up_3',
                           use_bias=False, strides=1, is_training=is_training, activation=relu6)
    inputs = tf.concat([route, inputs], axis=-1)
    c3 = inputs
    inputs = tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME')

    conv_out = _make_divisible(128 * cfgs.CSPNET_SHRINK)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out * 2, kernel_size=3, name='c4_1',
                           use_bias=False, strides=1, is_training=is_training, activation=hard_swish)
    route = inputs
    inputs = route_group(inputs, 2, 1)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out, kernel_size=3, name='c4_up_1',
                           use_bias=False, strides=1, is_training=is_training, activation=hard_swish)
    route_1 = inputs
    inputs = _conv_bn_relu(inputs, filters_num=conv_out, kernel_size=3, name='c4_up_2',
                           use_bias=False, strides=1, is_training=is_training, activation=hard_swish)
    inputs = tf.concat([inputs, route_1], axis=-1)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out * 2, kernel_size=3, name='c4_up_3',
                           use_bias=False, strides=1, is_training=is_training, activation=hard_swish)
    inputs = tf.concat([route, inputs], axis=-1)
    c4 = inputs
    inputs = tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME')

    conv_out = _make_divisible(256 * cfgs.CSPNET_SHRINK)
    inputs = _conv_bn_relu(inputs, filters_num=conv_out * 2, kernel_size=3, name='c5_1',
                           use_bias=False, strides=1, is_training=is_training, activation=hard_swish)
    c5 = inputs
    return c2, c3, c4, c5


def cspdarknet53_tiny_dwise(inputs, is_training):
    init_conv_out = _make_divisible(32 * cfgs.CSPNET_SHRINK)
    inputs = _conv_bn_relu(inputs, filters_num=init_conv_out, kernel_size=3, name='init',
                           use_bias=False, strides=2, is_training=is_training, activation=leaky_relu)

    conv_out = _make_divisible(64 * cfgs.CSPNET_SHRINK)
    route = inputs
    inputs = _dwise_bn_act(inputs, is_training, 'c2_0', leaky_relu)
    inputs = tf.concat([route, inputs], axis=-1)
    inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME')

    conv_out = _make_divisible(32 * cfgs.CSPNET_SHRINK)
    inputs = _dwise_bn_act(inputs, is_training, 'c2_1', leaky_relu)
    route = inputs
    inputs = route_group(inputs, 2, 1)
    inputs = _dwise_bn_act(inputs, is_training, 'c2_up_1', leaky_relu)
    route_1 = inputs
    inputs = _dwise_bn_act(inputs, is_training, 'c2_up_2', leaky_relu)
    inputs = tf.concat([inputs, route_1], axis=-1)
    inputs = _dwise_bn_act(inputs, is_training, 'c2_up_3', leaky_relu)
    inputs = tf.concat([route, inputs], axis=-1)
    c2 = inputs
    inputs = tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME')

    conv_out = _make_divisible(64 * cfgs.CSPNET_SHRINK)
    inputs = _dwise_bn_act(inputs, is_training, 'c3_1', leaky_relu)
    route = inputs
    inputs = route_group(inputs, 2, 1)
    inputs = _dwise_bn_act(inputs, is_training, 'c3_up_1', leaky_relu)
    route_1 = inputs
    inputs = _dwise_bn_act(inputs, is_training, 'c3_up_2', leaky_relu)
    inputs = tf.concat([inputs, route_1], axis=-1)
    inputs = _dwise_bn_act(inputs, is_training, 'c3_up_3', leaky_relu)
    inputs = tf.concat([route, inputs], axis=-1)
    c3 = inputs
    inputs = tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME')

    conv_out = _make_divisible(128 * cfgs.CSPNET_SHRINK)
    inputs = _dwise_bn_act(inputs, is_training, 'c4_1', leaky_relu)
    route = inputs
    inputs = route_group(inputs, 2, 1)
    inputs = _dwise_bn_act(inputs, is_training, 'c4_up_1', leaky_relu)
    route_1 = inputs
    inputs = _dwise_bn_act(inputs, is_training, 'c4_up_2', leaky_relu)
    inputs = tf.concat([inputs, route_1], axis=-1)
    inputs = _dwise_bn_act(inputs, is_training, 'c4_up_3', leaky_relu)
    inputs = tf.concat([route, inputs], axis=-1)
    c4 = inputs
    inputs = tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME')

    conv_out = _make_divisible(256 * cfgs.CSPNET_SHRINK)
    inputs = _dwise_bn_act(inputs, is_training, 'c5_1', leaky_relu)
    c5 = inputs
    return c2, c3, c4, c5


def cspdarknet53_tiny_dwise_focus(inputs, is_training):
    init_conv_out = _make_divisible(32 * cfgs.CSPNET_SHRINK)
    inputs = focus(inputs, init_conv_out, is_training)

    conv_out = _make_divisible(64 * cfgs.CSPNET_SHRINK)
    route = inputs
    inputs = _dwise_bn_act(inputs, is_training, 'c2_0', leaky_relu)
    inputs = tf.concat([route, inputs], axis=-1)
    inputs = tf.layers.max_pooling2d(inputs, 3, 2, padding='SAME')

    conv_out = _make_divisible(32 * cfgs.CSPNET_SHRINK)
    inputs = _dwise_bn_act(inputs, is_training, 'c2_1', leaky_relu)
    route = inputs
    inputs = route_group(inputs, 2, 1)
    inputs = _dwise_bn_act(inputs, is_training, 'c2_up_1', leaky_relu)
    route_1 = inputs
    inputs = _dwise_bn_act(inputs, is_training, 'c2_up_2', leaky_relu)
    inputs = tf.concat([inputs, route_1], axis=-1)
    inputs = _dwise_bn_act(inputs, is_training, 'c2_up_3', leaky_relu)
    inputs = tf.concat([route, inputs], axis=-1)
    c2 = inputs
    inputs = tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME')

    conv_out = _make_divisible(64 * cfgs.CSPNET_SHRINK)
    inputs = _dwise_bn_act(inputs, is_training, 'c3_1', leaky_relu)
    route = inputs
    inputs = route_group(inputs, 2, 1)
    inputs = _dwise_bn_act(inputs, is_training, 'c3_up_1', leaky_relu)
    route_1 = inputs
    inputs = _dwise_bn_act(inputs, is_training, 'c3_up_2', leaky_relu)
    inputs = tf.concat([inputs, route_1], axis=-1)
    inputs = _dwise_bn_act(inputs, is_training, 'c3_up_3', leaky_relu)
    inputs = tf.concat([route, inputs], axis=-1)
    c3 = inputs
    inputs = tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME')

    conv_out = _make_divisible(128 * cfgs.CSPNET_SHRINK)
    inputs = _dwise_bn_act(inputs, is_training, 'c4_1', leaky_relu)
    route = inputs
    inputs = route_group(inputs, 2, 1)
    inputs = _dwise_bn_act(inputs, is_training, 'c4_up_1', leaky_relu)
    route_1 = inputs
    inputs = _dwise_bn_act(inputs, is_training, 'c4_up_2', leaky_relu)
    inputs = tf.concat([inputs, route_1], axis=-1)
    inputs = _dwise_bn_act(inputs, is_training, 'c4_up_3', leaky_relu)
    inputs = tf.concat([route, inputs], axis=-1)
    c4 = inputs
    inputs = tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME')

    conv_out = _make_divisible(256 * cfgs.CSPNET_SHRINK)
    inputs = _dwise_bn_act(inputs, is_training, 'c5_1', leaky_relu)
    c5 = inputs
    return c2, c3, c4, c5
