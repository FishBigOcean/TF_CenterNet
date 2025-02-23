import tensorflow as tf
import cfgs

def _bn(inputs, is_training):
    bn = tf.layers.batch_normalization(
        inputs=inputs,
        training=is_training,
        momentum=cfgs.BN_MOMENTUM
    )
    return bn

def _conv(inputs, filters, kernel_size, strides=1, padding='same', activation=tf.nn.relu6, is_training=False, use_bn=True):
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

def upsampling(inputs,  method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        input_shape = tf.shape(inputs)
        output = tf.image.resize_nearest_neighbor(inputs, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
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
    return output

