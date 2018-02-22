import numpy as np

import keras
from keras.layers import Input, Reshape
from keras import optimizers
from keras.models import Model
from keras import losses
from keras import metrics

from models.vgg import build_VGG_Bnorm


# Select and build model
def getModel(mdl, actTrnData, actTrnLabels, tstData, tstLabels, class_number):

    inputData = Input(shape=(actTrnData.shape[1:]), name='data')

    # Model selector
    if mdl == "vgg1":
        net = build_VGG_Bnorm(inputData, block_channels=[64,96], block_layers=[2,2], fcChannels=[128,128], p_drop=0.5, classes=class_number)
    elif mdl == "vgg2":
        net = build_VGG_Bnorm(inputData, block_channels=[64,96,128], block_layers=[3,3,3], fcChannels=[512,512], p_drop=0.5, classes=class_number)
    else:
        print("Incorrect model")
        sys.exit()

    model = Model(inputs=[inputData], outputs=[net])
    model.summary()
    model.compile(
        loss=losses.categorical_crossentropy,
        optimizer=optimizers.Adam(lr=0.001),
        metrics=[metrics.categorical_accuracy])

    # Slight training on a small labeled set
    # How many epochs?
    model.fit(
        x=actTrnData, y=actTrnLabels,
        batch_size=48, epochs=3, verbose=1,
        validation_data=[tstData, tstLabels], shuffle=True)

    return model
