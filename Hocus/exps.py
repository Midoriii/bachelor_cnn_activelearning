from __future__ import print_function
import sys
import os.path


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np

from tools.tools import readCIFAR, mapLabelsOneHot,load_mnist
from tools.tools import collage

import keras
from keras.layers import Input, Reshape
from keras import optimizers
from keras.models import Model
from keras import losses
from keras import metrics
from keras.datasets import mnist

from models.vgg import *
from methods.uncertainty import *


# Acquire argument values
def getArgs():
    # Get arguments from console
    args = sys.argv[1:]

    # Args have to be dataset, model, method, percentage to label and outputfile
    if len(args) < 5:
        print("Needs arguments, usage: python exps.py (dataset) (model) (method) (percentage to label) (outputfile)")
        sys.exit()

    # Dataset
    dataset = args[0]
    # Network model
    mdl = args[1]
    # What amount of dataset to label
    percentage_limit = np.int(args[3])
    # Method to choose sample to be labeled
    mthd = args[2]
    # Ouput file path
    out = args[4]

    return dataset, mdl, mthd, percentage_limit, out


# Set up tensorflow and gpu usage
def tfSetUp():
    # just to use a fraction of GPU memory 
    # This is not needed on dedicated machines.
    # Allows you to share the GPU.
    # This is specific to tensorflow.
    gpu_memory_usage=0.7
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_usage
    set_session(tf.Session(config=config))
    return


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
        actPoolData = trnData[10000:100000,:,:,:]
        actPoolLabels = trnLabels[10000:100000,:]
        
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

        actPoolData = trnData[6000:60000,:,:,:]
        actPoolLabels = trnLabels[6000:60000,:]

    else:
        print("Incorrect dataset")
        sys.exit()
        
    return trnData, tstData, trnLabels, tstLabels, actTrnData, actTrnLabels, actPoolData, actPoolLabels


# Select and build model
def getModel(mdl, actTrnData, actTrnLabels, tstData, tstLabels):
    
    inputData = Input(shape=(actTrnData.shape[1:]), name='data')
    
    # Model selector
    if mdl == "vgg1":
        net = build_VGG_Bnorm(inputData, block_channels=[64,128], block_layers=[2,2], fcChannels=[256,256], p_drop=0.55, classes=10)
    elif mdl == "vgg2":
        net = build_VGG_Bnorm(inputData, block_channels=[64,128,256], block_layers=[3,3,3], fcChannels=[512,512], p_drop=0.5, classes=10)
    else:
        net = build_VGG_Bnorm(inputData, block_channels=[64,128,256,256], block_layers=[3,3,3,3], fcChannels=[512,512], p_drop=0.5, classes=10)
        
    model = Model(inputs=[inputData], outputs=[net])
    model.summary()
    model.compile(
        loss=losses.categorical_crossentropy, 
        optimizer=optimizers.Adam(lr=0.001), 
        metrics=[metrics.categorical_accuracy])

    # Slight training on a small labeled set
    model.fit(
        x=actTrnData, y=actTrnLabels,
        batch_size=48, epochs=3, verbose=1, 
        validation_data=[tstData, tstLabels], shuffle=True)

    return model


# Compute error rate manually, for graph projection
def getError(Model, Data, Labels):
    classProb = Model.predict(x=Data, verbose=1)
    correctProb = (classProb * Labels).sum(axis=1)
    wrongProb = (classProb * (1-Labels)).max(axis=1)

    accuracy = (correctProb > wrongProb).mean()
    error = 1.0 - accuracy
    print("\n acc: \n", accuracy)
    return error


# Active learning loop
def actLearning(model, mdl, mthd, percentage_limit, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels):
    # Initial error
    err = getError(model, tstData, tstLabels)
    # Initial labeled %
    labeled = [100*(float(len(actTrnData))/float(len(trnData)))]

    # While actively labeled percentage is lesser than the desired
    while max(labeled) < percentage_limit:
        # Add uncertain sample to actTrnData and its label to actTrnLabels
        ask_id = qs.querry(K)
        
        addition = actPoolData[ask_id]
        #addition = np.expand_dims(addition, axis=0)
        additionL = actPoolLabels[ask_id]
        #additionL = np.expand_dims(additionL, axis=0)
        
        actTrnData = np.append(actTrnData, addition, axis=0)
        actTrnLabels = np.append(actTrnLabels, additionL, axis=0)
        print("", actTrnData.shape)
        # Add high certainty samples temporarily TODOOOOOOOOOO
        # Train on the new training batch
        model.fit(
            x=actTrnData, y=actTrnLabels,
            batch_size=48, epochs=2, verbose=1, 
            validation_data=[tstData, tstLabels], shuffle=True)
        # Keep error/labeled %
        err = np.append(err, getError(model, tstData, tstLabels))
        labeled = np.append(labeled, 100*(float(len(actTrnData))/float(len(trnData))))
        
    # Plot percentages as graph
    #plt.plot(labeled, err, 'g')
    #plt.xlabel('Labeled %')
    #plt.ylabel('Error')
    #plt.title("" + mdl + "_" + mthd)
    #axes = plt.gca()
    #axes.set_xlim(10, max(labeled))
    #axes.set_ylim(0.0, 1.0)

    #plt.draw()

    # Save figure as .png
    #false_me = 1
    #n = 1
    # Loop to autoincrement fig number in filename
    #while false_me == 1 :
    #    figpath = mdl + "_" + mthd + "_" + str(n) + ".png"
    #    if os.path.isfile(figpath):
    #        n = n + 1
    #    else:
    #        plt.savefig(figpath)
    #        false_me = 0
   
    print("",err)
    print("",labeled)
    return err, labeled
    

def main():
    # Get model, method and number of querries
    dataset, mdl, mthd, percentage_limit, outfile = getArgs()
    # Set up tensorflow
    tfSetUp()
    # Read dataset
    trnData, tstData, trnLabels, tstLabels, actTrnData, actTrnLabels, actPoolData, actPoolLabels = getData(dataset)

    # Number of samples to label in one go (1%)
    K = len(trnData)/100
    
    model = getModel(mdl, actTrnData, actTrnLabels, tstData, tstLabels)
    # Save weights after model creation and slight training
    model.save_weights('tmpweights.h5')

    # Error definition for methods
    lcErr = 0
    smErr = 0
    entErr = 0
    labeled = 0

    # Run active learning with only one method or compare several
    # TODO: remove unnecessary arguments, move plotting graphs to another script, make active learning run in a while loop until reaching desired labeled %
    if mthd == "lc" or mthd == "all":
        qs = UncertaintySampling(model, "lc", actPoolData)
        lcErr, labeled = actLearning(model, mdl, "lc", percentage_limit, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "sm" or mthd == "all":
        qs = UncertaintySampling(model, "sm", actPoolData)
        smErr, labeled = actLearning(model, mdl, "sm", percentage_limit, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "ent" or mthd == "all":
        qs = UncertaintySampling(model, "ent", actPoolData)
        entErr, labeled = actLearning(model, mdl, "ent", percentage_limit, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    # Save data to a npz file
    np.savez(outfile, labeled=labeled, lc=lcErr, sm=smErr, ent=entErr, dset=dataset, model=mdl, method=mthd)
    
if __name__ == '__main__':
    main()
