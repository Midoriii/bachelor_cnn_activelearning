import numpy as np

import h5py
import keras
import sys
from keras.datasets import mnist

from tools.tools import readCIFAR, mapLabelsOneHot,load_mnist
from tools.tools import collage


# Read image data and labels
def getData(dataset):
    if dataset == "cifar":
        # This reads the dataset
        trnData, tstData, trnLabels, tstLabels = readCIFAR('../data/cifar-10-batches-py')

        # Convert categorical labels to one-hot encoding which
        # is needed by categorical_crossentropy in Keras.
        # This is not universal. The loss can be easily implemented
        # with category IDs as labels.
        trnLabels = mapLabelsOneHot(trnLabels)
        tstLabels = mapLabelsOneHot(tstLabels)

        trnData = trnData.astype(np.float32) / 255.0 - 0.5
        tstData = tstData.astype(np.float32) / 255.0 - 0.5

        # Small "labeled" batch for initial weight training
        actTrnData = trnData[:10000,:,:,:]
        actTrnLabels = trnLabels[:10000,:]

        # Pool from which to choose "unlabeled" samples
        actPoolData = trnData[10000:,:,:,:]
        actPoolLabels = trnLabels[10000:,:]

    elif dataset == "fashion":
        # URLs for a function to download .gz from
        train_images_path = keras.utils.get_file('train-images-idx3-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz')
        train_labels_path = keras.utils.get_file('train-labels-idx1-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz')
        test_images_path = keras.utils.get_file('t10k-images-idx3-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz')
        test_labels_path = keras.utils.get_file('t10k-labels-idx1-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz')

        trnData, trnLabels = load_mnist(train_images_path, train_labels_path)
        tstData, tstLabels = load_mnist(test_images_path, test_labels_path)

        trnData = trnData.astype(np.float32) / 255.0 - 0.5
        tstData = tstData.astype(np.float32) / 255.0 - 0.5

        trnLabels = mapLabelsOneHot(trnLabels)
        tstLabels = mapLabelsOneHot(tstLabels)

        trnData = trnData.reshape(-1, 28, 28, 1)
        tstData = tstData.reshape(-1, 28, 28, 1)

        actTrnData = trnData[:6000,:,:,:]
        actTrnLabels = trnLabels[:6000,:]

        actPoolData = trnData[6000:,:,:,:]
        actPoolLabels = trnLabels[6000:,:]

    elif dataset == "imnet":
        hdf5_file = h5py.File('dataset/imnet/imnet.hdf5', "r")

        trnData = hdf5_file["train_img"][:, ...]
        tstData = hdf5_file["tst_img"][:, ...]

        trnLabels = hdf5_file["train_labels"][:]
        tstLabels = hdf5_file["tst_labels"][:]

        trnData = trnData.astype(np.float32) / 255.0 - 0.5
        tstData = tstData.astype(np.float32) / 255.0 - 0.5

        trnLabels = mapLabelsOneHot(trnLabels)
        tstLabels = mapLabelsOneHot(tstLabels)

        actTrnData = trnData[:int(0.11*len(trnData)),:,:,:]
        actTrnLabels = trnLabels[:int(0.11*len(trnData)),:]

        actPoolData = trnData[int(0.11*len(trnData)):,:,:,:]
        actPoolLabels = trnLabels[int(0.11*len(trnData)):,:]

    elif dataset == "vgg2f":
        hdf5_file = h5py.File('dataset/vgg2f/vgg2f.hdf5', "r")

        trnData = hdf5_file["train_img"][:, ...]
        tstData = hdf5_file["tst_img"][:, ...]

        trnLabels = hdf5_file["train_labels"][:]
        tstLabels = hdf5_file["tst_labels"][:]

        trnData = trnData.astype(np.float32) / 255.0 - 0.5
        tstData = tstData.astype(np.float32) / 255.0 - 0.5

        trnLabels = mapLabelsOneHot(trnLabels)
        tstLabels = mapLabelsOneHot(tstLabels)

        actTrnData = trnData[:int(0.11*len(trnData)),:,:,:]
        actTrnLabels = trnLabels[:int(0.11*len(trnData)),:]

        actPoolData = trnData[int(0.11*len(trnData)):,:,:,:]
        actPoolLabels = trnLabels[int(0.11*len(trnData)):,:]

    else:
        print("Incorrect dataset")
        sys.exit()

    # Get the number of classes
    class_number = int(tstLabels.shape[1])

    return trnData, tstData, trnLabels, tstLabels, actTrnData, actTrnLabels, actPoolData, actPoolLabels, class_number
