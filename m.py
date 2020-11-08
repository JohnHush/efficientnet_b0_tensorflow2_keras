import tensorflow as tf
import collections

BlockSpec = collections.namedtuple( 'BlockSpec',
                                    [ 'input_nf',
                                      'expand_ratio',
                                      'dw_kernel_size',
                                      'dw_strides',
                                      'output_nf',
                                      'dropout_rate',
                                      'skip_conn'])

def MobileNetV2Block( inputs, block_spec, prefix='' ):
    # feature expansion
    # in the original paper, expand_ratio is set to 6
    expand_nf = block_spec.input_nf * block_spec.expand_ratio
    kernel_initializer = tf.keras.initializers.VarianceScaling( scale=2.0,
                                                                mode='fan_out',
                                                                distribution='normal')

    x = tf.keras.layers.Conv2D( expand_nf, 1,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                name=prefix + 'expanding_conv'
                                )(inputs)
    x = tf.keras.layers.BatchNormalization(name=prefix + 'expanding_bn')(x)
    x = tf.keras.activations.swish(x)

    # depth-wise convolution to generate higher-level features
    x = tf.keras.layers.DepthwiseConv2D( block_spec.dw_kernel_size,
                                         strides=block_spec.dw_strides,
                                         padding='same',
                                         use_bias=False,
                                         depthwise_initializer=kernel_initializer,
                                         name=prefix + 'dw_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + 'dw_bn')(x)
    x = tf.keras.activations.swish(x)

    # linear transform part in MobileNet V2 block,
    # there's no activation function in this part
    # when the stride of the depthwise CONV equals 2, there's no skip connection

    x = tf.keras.layers.Conv2D( block_spec.output_nf, 1,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                name=prefix + 'projecting_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + 'projecting_bn')(x)

    if block_spec.dropout_rate:
        # set of noise shape refer to Tensorflow API Guide
        x = tf.keras.layers.Dropout( rate=block_spec.dropout_rate,
                                     noise_shape=[None,1,1,1],
                                     name=prefix + 'dropout')(x)

    if block_spec.skip_conn:
        # the input_nf must equal to output_nf
        # for the ADD operator
        x = tf.keras.layers.Add( name=prefix + 'skip_conn')([x, inputs])

    return x


def test():

    x = tf.random.normal(shape=[5,4,4])

    x = tf.keras.layers.Dropout( 0.2 , noise_shape=[None,1,1])(x, training=True )
    print(x)

test()

