import numpy as np

import h5py
import keras
import sys
from keras.datasets import mnist

from tools.tools import readCIFAR, mapLabelsOneHot,load_mnist
from tools.tools import collage


# Read image data and labels
def getData(dataset):

    hdf5_file = h5py.File(dataset, "r")

    trnData = hdf5_file["train_img"][:, ...]
    tstData = hdf5_file["tst_img"][:, ...]

    trnLabels = hdf5_file["train_labels"][:]
    tstLabels = hdf5_file["tst_labels"][:]

    trnData = trnData.astype(np.float32) / 255.0 - 0.5
    tstData = tstData.astype(np.float32) / 255.0 - 0.5

    trnLabels = mapLabelsOneHot(trnLabels)
    tstLabels = mapLabelsOneHot(tstLabels)

    actTrnData = trnData[:int(0.10*len(trnData)),:,:,:]
    actTrnLabels = trnLabels[:int(0.10*len(trnData)),:]

    actPoolData = trnData[int(0.10*len(trnData)):,:,:,:]
    actPoolLabels = trnLabels[int(0.10*len(trnData)):,:]

    # Get the number of classes
    class_number = int(tstLabels.shape[1])

    return trnData, tstData, trnLabels, tstLabels, actTrnData, actTrnLabels, actPoolData, actPoolLabels, class_number
