from keras.layers import Input, Reshape, Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, Conv2D, MaxPooling2D, PReLU
from keras.models import Model
from keras import regularizers

def build_VGG_Bnorm_block(net, channels, layers, prefix):
    for i in range(layers):
        net = Conv2D(channels, 3, padding='same',
                    name='{}.{}'.format(prefix, i))(net)
        net = BatchNormalization()(net)
        net = PReLU()(net)
    net = MaxPooling2D(2, 2, padding="same")(net)
    return net

def build_VGG_Bnorm(input_data, block_channels=[16,32,64], block_layers=[2,2,2], fcChannels=[256,256], p_drop=0.4, classes=10):
    net = input_data
    for i, (cCount, lCount) in enumerate(zip(block_channels, block_layers)):
        net = build_VGG_Bnorm_block(net, cCount, lCount, 'conv{}'.format(i))
        net = Dropout(rate=0.25)(net)
        
    net = Flatten()(net)
    
    for i, cCount in enumerate(fcChannels):
        net = Dense(cCount, name='fc{}'.format(i))(net)
        net = BatchNormalization()(net)
        net = PReLU()(net)

        net = Dropout(rate=p_drop)(net)
    
    net = Dense(classes, name='out', activation='softmax')(net)

    return net
