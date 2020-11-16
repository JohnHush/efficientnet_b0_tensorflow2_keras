import tensorflow as tf
import collections
import math
from tensorflow.keras.layers.experimental import preprocessing

BlockSpec = collections.namedtuple( 'BlockSpec',
                                    [ 'input_nf',
                                      'output_nf',
                                      'expand_ratio',
                                      'dw_kernel_size',
                                      'dw_strides',
                                      'se_ratio',
                                      'drop_conn_rate',
                                      'skip_conn',
                                      'repeat_num',
                                      'activation',
                                      'prefix'])
BlockSpec.__new__.__defaults__ = (None,) * len(BlockSpec._fields)

NNSpec = collections.namedtuple( 'NNSpec',
                                 ['w',
                                  'd',
                                  'depth_divisor',
                                  'include_top',
                                  'dropout_rate',
                                  'classes',
                                  'activation',
                                  'drop_conn_rate',
                                  'pooling'])
NNSpec.__new__.__defaults__ = (None,) * len(NNSpec._fields)

def mobilenet_v2_block( inputs, block_spec:BlockSpec ):
    conv_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                    mode='fan_out',
                                                                    distribution='normal')
    dense_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1./3,
                                                                     mode='fan_out',
                                                                     distribution='uniform')
    # feature expanding
    expand_nf = block_spec.input_nf * block_spec.expand_ratio
    if block_spec.expand_ratio != 1:
        x = tf.keras.layers.Conv2D( expand_nf,
                                    1,
                                    padding='same',
                                    use_bias=False,
                                    kernel_initializer=conv_kernel_initializer,
                                    name=block_spec.prefix + 'expanding_conv')(inputs)
        x = tf.keras.layers.BatchNormalization(name=block_spec.prefix + 'expanding_bn')(x)
        x = tf.keras.layers.Activation(block_spec.activation, name=block_spec.prefix + 'expanding_nonlinear')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = tf.keras.layers.DepthwiseConv2D( block_spec.dw_kernel_size,
                                         block_spec.dw_strides,
                                         padding='same',
                                         use_bias=False,
                                         depthwise_initializer=conv_kernel_initializer,
                                         name=block_spec.prefix + 'dw_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=block_spec.prefix + 'dw_bn')(x)
    x = tf.keras.layers.Activation(block_spec.activation, name=block_spec.prefix + 'dw_nonlinear')(x)

    # Squeeze and Excitation Module
    if 0 < block_spec.se_ratio < 1.:
        se_nf = max( 1, int(block_spec.input_nf * block_spec.se_ratio) )
        se = tf.keras.layers.GlobalAveragePooling2D(name=block_spec.prefix + 'se_pooling')(x)
        se = tf.keras.layers.Reshape(target_shape=[1,1,expand_nf], name=block_spec.prefix + 'se_reshape')(se)
        se = tf.keras.layers.Conv2D( se_nf,
                                     1,
                                     padding='same',
                                     activation=block_spec.activation,
                                     kernel_initializer=conv_kernel_initializer,
                                     name=block_spec.prefix + 'se_reduce')(se)
        se = tf.keras.layers.Conv2D( expand_nf,
                                     1,
                                     padding='same',
                                     activation='sigmoid',
                                     kernel_initializer=conv_kernel_initializer,
                                     name=block_spec.prefix + 'se_expand')(se)
        x = tf.keras.layers.Multiply(name=block_spec.prefix + 'se_multiply')([se, x])

    # projecting step
    x = tf.keras.layers.Conv2D( block_spec.output_nf,
                                1,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=conv_kernel_initializer,
                                name=block_spec.prefix + 'projecting_conv'
                                )(x)
    x = tf.keras.layers.BatchNormalization(name=block_spec.prefix + 'projecting_bn')(x)
    if block_spec.skip_conn and block_spec.dw_strides==1 and block_spec.input_nf==block_spec.output_nf:
        if block_spec.drop_conn_rate > 0:
            x = tf.keras.layers.Dropout(block_spec.drop_conn_rate,
                                        noise_shape=[None,1,1,1],
                                        name=block_spec.prefix + 'drop_conn')(x)

        x = tf.keras.layers.Add(name=block_spec.prefix + 'add')([x, inputs])

    return x

class MobileV2Block( tf.keras.layers.Layer ):
    # block designed in MobileNetV2
    # inverted Residual Bottleneck Block
    # a squeeze-excitation block could also been added
    def __init__(self, block_spec, prefix='', **kwargs ):
        super(MobileV2Block, self).__init__(**kwargs)
        self._cfg = block_spec
        self._prefix = prefix

    def build(self, input_shape):
        # feature expanding
        # expand_ratio set to 6 in the original paper
        self._expand_nf = self._cfg.input_nf * self._cfg.expand_ratio
        self._kernel_initializer = tf.keras.initializers.VarianceScaling( scale=2.0,
                                                                          mode='fan_out',
                                                                          distribution='normal')
        # self._activation = tf.keras.activations.swish

        self._expanding_conv = tf.keras.layers.Conv2D( self._expand_nf, 1,
                                                       padding='same',
                                                       use_bias=False,
                                                       kernel_initializer=self._kernel_initializer,
                                                       name=self._prefix + 'expanding_conv'
                                                       )
        self._expanding_bn = tf.keras.layers.BatchNormalization(name=self._prefix + 'expanding_bn')
        self._expanding_swish = SWISH(name=self._prefix + 'expanding_swish')

        # self._expanding_nonlinear = tf.nn.swish(name=self._prefix + 'expanding_nonlinear')

        # depth-wise convolution to generate higher-level features
        self._dw_conv = tf.keras.layers.DepthwiseConv2D( self._cfg.dw_kernel_size,
                                                         strides=self._cfg.dw_strides,
                                                         padding='same',
                                                         use_bias=False,
                                                         depthwise_initializer=self._kernel_initializer,
                                                         name=self._prefix + 'dw_conv')
        self._dw_bn = tf.keras.layers.BatchNormalization(name=self._prefix + 'dw_bn')
        self._dw_swish = SWISH(name=self._prefix + 'dw_swish')

        if self._cfg.se_ratio:
            self._squeeze_nf = max( 1, self._cfg.input_nf * self._cfg.se_ratio )

            self._se_squeeze = tf.keras.layers.GlobalAveragePooling2D(name=self._prefix + 'se_squeeze')
            self._se_reshape = tf.keras.layers.Reshape(target_shape=(1, 1, self._expand_nf),
                                                       name=self._prefix + 'se_reshape')
            self._se_reduce = tf.keras.layers.Conv2D(self._squeeze_nf, 1,
                                                     activation='swish',
                                                     padding='same',
                                                     use_bias=True,
                                                     kernel_initializer=self._kernel_initializer,
                                                     name=self._prefix + 'se_reduce')
            self._se_expand = tf.keras.layers.Conv2D(self._expand_nf, 1,
                                                     activation='sigmoid',
                                                     padding='same',
                                                     use_bias=True,
                                                     kernel_initializer=self._kernel_initializer,
                                                     name=self._prefix + 'se_expand')

            self._se_multiply = tf.keras.layers.Multiply(name=self._prefix + 'se_excite')

        # linear transform part in MobileNet V2 block,
        # there's no activation function in this part
        # when the stride of the depthwise CONV equals 2, there's no skip connection

        self._projecting_conv = tf.keras.layers.Conv2D(self._cfg.output_nf, 1,
                                                       padding='same',
                                                       use_bias=False,
                                                       kernel_initializer=self._kernel_initializer,
                                                       name=self._prefix + 'projecting_conv')

        self._projecting_bn = tf.keras.layers.BatchNormalization(name=self._prefix + 'projecting_bn')

        if self._cfg.skip_conn and self._cfg.input_nf==self._cfg.output_nf:
            if self._cfg.dropout_rate:
                # set of noise shape refer to Tensorflow API Guide
                self._dropout = tf.keras.layers.Dropout( rate=self._cfg.dropout_rate,
                                                         noise_shape=[None,1,1,1],
                                                         name=self._prefix + 'dropout')

            # the input_nf must equal to output_nf
            # for the ADD operator
            self._skip_conn = tf.keras.layers.Add(name=self._prefix + 'skip_conn')

    def call(self, inputs, **kwargs):
        x = self._expanding_conv(inputs)
        x = self._expanding_bn(x)
        x = self._expanding_swish(x)

        x = self._dw_conv(x)
        x = self._dw_bn(x)
        x = self._dw_swish(x)

        if self._cfg.se_ratio:
            se = self._se_squeeze(x)
            se = self._se_reshape(se)
            se = self._se_reduce(se)
            se = self._se_expand(se)
            x  = self._se_multiply([se, x])

        x = self._projecting_conv(x)
        x = self._projecting_bn(x)

        if self._cfg.skip_conn and self._cfg.input_nf==self._cfg.output_nf:
            if self._cfg.dropout_rate:
                x = self._dropout(x)

            x = self._skip_conn([inputs, x ])

        return x

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))

class EfficientNet(tf.keras.Model):
    def __init__(self, block_specs, nn_spec ):
        super(EfficientNet, self).__init__()
        self._building_blocks = []
        self._cfg = block_specs
        self._nn_spec = nn_spec

        self._kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                         mode='fan_out',
                                                                         distribution='normal')
        self._dense_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1./3,
                                                                               mode='fan_out',
                                                                               distribution='uniform')

        # cnn stem
        self._stem_conv = tf.keras.layers.Conv2D(round_filters(32, nn_spec.w, nn_spec.depth_divisor),
                                                 kernel_size=3,
                                                 strides=2,
                                                 kernel_initializer=self._kernel_initializer,
                                                 padding='same',
                                                 use_bias=False,
                                                 name='stem_conv')
        self._stem_bn = tf.keras.layers.BatchNormalization(name='stem_bn')
        self._stem_nonlinear = tf.keras.layers.Activation(nn_spec.activation, name='stem_nonlinear')

        # Several stages of MobileNetV2 blocks
        block_num_sum = sum( round_repeats(block_spec.repeat_num, nn_spec.d) for block_spec in block_specs ) *1.
        block_index = 0

        for block_stage, block_spec in enumerate(block_specs):
            block_spec = block_spec._replace(
                input_nf=round_filters(block_spec.input_nf, nn_spec.w, nn_spec.depth_divisor),
                output_nf=round_filters(block_spec.output_nf, nn_spec.w, nn_spec.depth_divisor),
                repeat_num=round_repeats(block_spec.repeat_num, nn_spec.d )
            )

            for sub_block_index in range(block_spec.repeat_num):
                block_spec = block_spec._replace( drop_conn_rate=nn_spec.drop_conn_rate * block_index / block_num_sum,
                                                  prefix='block{}{}_'.format(block_stage, chr(sub_block_index+97) ))
                if sub_block_index > 0:
                    block_spec = block_spec._replace(
                        dw_strides=1,
                        input_nf=block_spec.output_nf
                    )
                self._building_blocks.append(block_spec)
                block_index += 1

        # cnn head
        self._head_conv = tf.keras.layers.Conv2D(round_filters(1280, nn_spec.w, nn_spec.depth_divisor),
                                                 kernel_size=1,
                                                 strides=1,
                                                 kernel_initializer=self._kernel_initializer,
                                                 padding='same',
                                                 use_bias=False,
                                                 name='head_conv')
        self._head_bn = tf.keras.layers.BatchNormalization(name='head_bn')
        self._head_nonlinear = tf.keras.layers.Activation(nn_spec.activation, name='head_nonlinear')

        if nn_spec.include_top:
            self._top_pooling = tf.keras.layers.GlobalAveragePooling2D(name='top_pooling')
            if nn_spec.dropout_rate > 0:
                self._top_dropout = tf.keras.layers.Dropout(nn_spec.dropout_rate, name='top_dropout')

            self._top_fc = tf.keras.layers.Dense( nn_spec.classes,
                                                  activation='softmax',
                                                  kernel_initializer=self._dense_kernel_initializer,
                                                  name='probs')
        else:
            if nn_spec.pooling == 'avg':
                self._top_pooling = tf.keras.layers.GlobalAveragePooling2D(name='top_pooling')
            elif nn_spec.pooling == 'max':
                self._top_pooling = tf.keras.layers.GlobalMaxPool2D(name='top_pooling')
            else:
                pass

        # print( self._building_blocks )

        # self._xx = tf.keras.layers.Activation('relu')
        # self._yy = tf.keras.layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        # stem
        x = self._stem_conv(inputs)
        x = self._stem_bn(x)
        x = self._stem_nonlinear(x)

        # for i in self._building_blocks:
        #     print(i)
        # main mobile blocks
        for block_spec in self._building_blocks:
        #     print('hhh')
            x = mobilenet_v2_block(x, block_spec)
            # x = tf.keras.layers.Activation('relu')(x)
        # x = tf.keras.layers.Activation('relu')(x)
        # x = tf.keras.layers.Activation('relu')(x)
        # x = tf.keras.layers.Dense(10)(x)
        #
        # x = self._xx(x)
        #
        # x = self._yy(x)
        # x = self._yy(x)
        #
        # top conv-bn-nonlinear
        x = self._head_conv(x)
        x = self._head_bn(x)
        x = self._head_nonlinear(x)

        if self._nn_spec.include_top:
            x = self._top_pooling(x)
            if self._nn_spec.dropout_rate > 0:
                x = self._top_dropout(x)
            x = self._top_fc(x)
        else:
            if self._nn_spec.pooling =='avg' or self._nn_spec.pooling =='max':
                x = self._top_pooling(x)

        return x

def build_efficient_net_model( block_specs, nn_spec ):
    import tensorflow.keras.layers as L
    _kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                mode='fan_out',
                                                                distribution='normal')
    _dense_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1. / 3,
                                                                      mode='fan_out',
                                                                      distribution='uniform')

    inputs = tf.keras.Input(shape=(224,224,3))

    x = preprocessing.Rescaling(1./255)(inputs)
    x = preprocessing.Normalization()(x)

    x = L.Conv2D(round_filters(32, nn_spec.w, nn_spec.depth_divisor),
                 kernel_size=3,
                 strides=2,
                 kernel_initializer=_kernel_initializer,
                 padding='same',
                 use_bias=False,
                 name='stem_conv')(x)
    x = L.BatchNormalization(name='stem_bn')(x)
    x = L.Activation(nn_spec.activation, name='stem_nonlinear')(x)

    # Several stages of MobileNetV2 blocks
    block_num_sum = sum( round_repeats(block_spec.repeat_num, nn_spec.d) for block_spec in block_specs ) *1.
    block_index = 0

    for block_stage, block_spec in enumerate(block_specs):
        block_spec = block_spec._replace(
            input_nf=round_filters(block_spec.input_nf, nn_spec.w, nn_spec.depth_divisor),
            output_nf=round_filters(block_spec.output_nf, nn_spec.w, nn_spec.depth_divisor),
            repeat_num=round_repeats(block_spec.repeat_num, nn_spec.d )
        )

        for sub_block_index in range(block_spec.repeat_num):
            block_spec = block_spec._replace( drop_conn_rate=nn_spec.drop_conn_rate * block_index / block_num_sum,
                                              prefix='block{}{}_'.format(block_stage+1, chr(sub_block_index+97) ))
            if sub_block_index > 0:
                block_spec = block_spec._replace(
                    dw_strides=1,
                    input_nf=block_spec.output_nf
                )
            x = mobilenet_v2_block( x, block_spec )
            # self._building_blocks.append(block_spec)
            block_index += 1

    # cnn head
    x = tf.keras.layers.Conv2D(round_filters(1280, nn_spec.w, nn_spec.depth_divisor),
                               kernel_size=1,
                               strides=1,
                               kernel_initializer=_kernel_initializer,
                               padding='same',
                               use_bias=False,
                               name='head_conv')(x)
    x = L.BatchNormalization(name='head_bn')(x)
    x = L.Activation(nn_spec.activation, name='head_nonlinear')(x)

    if nn_spec.include_top:
        x = L.GlobalAveragePooling2D(name='top_pooling')(x)
        if nn_spec.dropout_rate > 0:
            x = L.Dropout(nn_spec.dropout_rate, name='top_dropout')(x)

        x = tf.keras.layers.Dense( nn_spec.classes,
                                   activation='softmax',
                                   kernel_initializer=_dense_kernel_initializer,
                                   name='probs')(x)
    else:
        if nn_spec.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D(name='top_pooling')(x)
        elif nn_spec.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPool2D(name='top_pooling')(x)
        else:
            pass

    model = tf.keras.Model(inputs=[inputs], outputs=[x])

    return model


def test_efficientnet_loadweight():
    DEFAULT_BLOCKS_ARGS = [
        BlockSpec(dw_kernel_size=3, repeat_num=1, input_nf=32, output_nf=16,
                  expand_ratio=1, skip_conn=True, dw_strides=1, se_ratio=0.25, activation='swish'),
        BlockSpec(dw_kernel_size=3, repeat_num=2, input_nf=16, output_nf=24,
                  expand_ratio=6, skip_conn=True, dw_strides=2, se_ratio=0.25, activation='swish'),
        BlockSpec(dw_kernel_size=5, repeat_num=2, input_nf=24, output_nf=40,
                  expand_ratio=6, skip_conn=True, dw_strides=2, se_ratio=0.25, activation='swish'),
        BlockSpec(dw_kernel_size=3, repeat_num=3, input_nf=40, output_nf=80,
                  expand_ratio=6, skip_conn=True, dw_strides=2, se_ratio=0.25, activation='swish'),
        BlockSpec(dw_kernel_size=5, repeat_num=3, input_nf=80, output_nf=112,
                  expand_ratio=6, skip_conn=True, dw_strides=1, se_ratio=0.25, activation='swish'),
        BlockSpec(dw_kernel_size=5, repeat_num=4, input_nf=112, output_nf=192,
                  expand_ratio=6, skip_conn=True, dw_strides=2, se_ratio=0.25, activation='swish'),
        BlockSpec(dw_kernel_size=3, repeat_num=1, input_nf=192, output_nf=320,
                  expand_ratio=6, skip_conn=True, dw_strides=1, se_ratio=0.25, activation='swish') ]

    NNSPEC = NNSpec( w=1.,
                     d=1.,
                     depth_divisor=8,
                     include_top=True,
                     dropout_rate=0.2,
                     activation='swish',
                     drop_conn_rate=0.2,
                     classes=1000 )
    BASE_WEIGHTS_PATH = 'https://storage.googleapis.com/keras-applications/'

    WEIGHTS_HASHES = {
        'b0': ('902e53a9f72be733fc0bcb005b3ebbac',
               '50bc09e76180e00e4465e1a485ddc09d'),
        'b1': ('1d254153d4ab51201f1646940f018540',
               '74c4e6b3e1f6a1eea24c589628592432'),
        'b2': ('b15cce36ff4dcbd00b6dd88e7857a6ad',
               '111f8e2ac8aa800a7a99e3239f7bfb39'),
        'b3': ('ffd1fdc53d0ce67064dc6a9c7960ede0',
               'af6d107764bb5b1abb91932881670226'),
        'b4': ('18c95ad55216b8f92d7e70b3a046e2fc',
               'ebc24e6d6c33eaebbd558eafbeedf1ba'),
        'b5': ('ace28f2a6363774853a83a0b21b9421a',
               '38879255a25d3c92d5e44e04ae6cec6f'),
        'b6': ('165f6e37dce68623721b423839de8be5',
               '9ecce42647a20130c1f39a5d4cb75743'),
        'b7': ('8c03f828fec3ef71311cd463b6759d99',
               'cbcfe4450ddf6f3ad90b1b398090fe4a'),
    }

    # model = EfficientNet( DEFAULT_BLOCKS_ARGS, NNSPEC )

    weight_path = tf.keras.utils.get_file( 'efficientnetb0.h5',
                                           BASE_WEIGHTS_PATH + 'efficientnetb0.h5',
                                           cache_subdir='modelsss',
                                           file_hash='902e53a9f72be733fc0bcb005b3ebbac')

    # print(weight_path)
    model = build_efficient_net_model( DEFAULT_BLOCKS_ARGS, NNSPEC )
    model.build( input_shape=(None,224,224,3) )

    load_status = model.load_weights(weight_path)
    # model.summary()
    print( load_status )
    tf.keras.utils.plot_model(model, '../bp.png', show_shapes=True, expand_nested=True)

    # n = 0
    # for layer in model.layers:
    #     if len(layer.weights) != 0:
    #         n += 1
    #
    # print(n)
    from skimage.io import imread
    import matplotlib.pyplot as plt
    # img = imread('model/panda.jpg')
    img = tf.io.gfile.GFile('model/panda.jpg', mode='rb' ).read()
    # plt.figure()
    # plt.imshow(img)
    # plt.show()

    from model.util import preprocess_for_eval
    img = preprocess_for_eval(img)
    print(img.shape)
    img = tf.expand_dims(img, 0)
    y = model.predict(img)

    r = tf.keras.applications.efficientnet.decode_predictions( y, 5 )
    print(r)

    # print(y)

def test_keras_application_efficientb0():
    model = tf.keras.applications.EfficientNetB0()

    tf.keras.utils.plot_model( model , 'bbbb.png', show_shapes=True )


    print( len(model.layers) )
    # for layer in model.layers:

    n = 0
    for layer in model.layers:
        if len(layer.weights) != 0:
            n += 1
            print( layer.name )

    print(n)



def test_mobilev2block():
    blockspec = BlockSpec(input_nf=32,
                          expand_ratio=6,
                          dw_kernel_size=3,
                          dw_strides=1,
                          se_ratio=0.25,
                          output_nf=16,
                          dropout_rate=0.2,
                          skip_conn=True )
    mobilev2b = MobileV2Block(block_spec=blockspec, prefix='fi_')
    # mobilev2b.build(input_shape=[112,112,32])


    input = tf.random.normal(shape=[2, 112, 112, 32])

    a = mobilev2b(input)
    mobilev2b.summary()

    inputs = tf.keras.Input( shape=[112,112,32] )

    outputs = mobilev2b(inputs, training=True)
    tf.keras.utils.plot_model( mobilev2b, 'pp.png', show_shapes=True, expand_nested=True )

    model = tf.keras.Model( inputs=[inputs], outputs=[outputs])

    # model.summary()

    # tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True, expand_nested=True)

def test_trace_mobilev2block():
    blockspec = BlockSpec(input_nf=32,
                          expand_ratio=6,
                          dw_kernel_size=3,
                          dw_strides=1,
                          se_ratio=0.25,
                          output_nf=32,
                          dropout_rate=0.2,
                          skip_conn=True)
    mobilev2b = MobileV2Block(block_spec=blockspec, prefix='fi_')

    @tf.function
    def traceme(x):
        return mobilev2b(x)

    writer = tf.summary.create_file_writer('log')
    tf.summary.trace_on(graph=True, profiler=True)

    traceme(tf.zeros( (2,112,112,32) ))

    with writer.as_default():
        tf.summary.trace_export(name='trace_model', step=0, profiler_outdir='log')


def test_dropout():
    class drooo( tf.keras.layers.Layer ):
        def __init__(self):
            super(drooo, self).__init__()
            self.b = tf.keras.layers.Dropout(0.5, noise_shape=[None,1])
        def call(self, inputs, **kwargs):
            return self.b(inputs)

    dlayer = drooo()
    x = tf.random.normal(shape=[4,4])
    print(x)
    x = dlayer(x, training=True)
    print(x)

def test_dropout2():
    class droo( tf.keras.Model ):
        def __init__(self, units):
            super(droo, self).__init__()
            self._units = units
            self._conv = tf.keras.layers.Dense(units, use_bias=False)
            self._drop = tf.keras.layers.Dropout(0.8)

        def call(self, inputs, training=None, mask=None):
            x = self._conv(inputs)
            x = self._drop(x)
            return x

    dl = droo(10)
    x= tf.random.normal(shape=[1,5])
    print( x )
    x = dl(x, training=True )
    print(x)

def test():
    l = []
    b = BlockSpec(input_nf=2)
    l.append(b)

    b = b._replace(input_nf=33)
    l.append(b)
    # print(b)

    for i in l:
        print(i)

# test()
# test_keras_application_efficientb0()
test_efficientnet_loadweight()
# test_dropout2()
# test_dropout()
# test_mobilev2block()
# test_trace_mobilev2block()
# test()
