from __future__ import print_function
import sys
import os.path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np

import keras
from keras.layers import Input, Reshape
from keras import optimizers
from keras.models import Model
from keras import losses
from keras import metrics
from keras.datasets import mnist

from models.model_selector import getModel

from dataset.dataset_selector import getData

from methods.uncertainty import UncertaintySampling
from methods.random_sampling import RandomSampling

from random import randint
from tools.tools import mapLabelsOneHot


# Acquire argument values
def getArgs():
    # Get arguments from console
    args = sys.argv[1:]

    # Args have to be dataset, model, method, percentage to label and outputfile
    if len(args) < 6:
        print("Needs arguments, usage: python exps.py (dataset) (model) (method) (percentage to label) (percentage of dataset to add in each iteration) (oracle correctness percentage) (outputfile)")
        sys.exit()

    # Dataset
    dataset = args[0]
    # Network model
    mdl = args[1]
    # What amount of dataset to label
    percentage_limit = np.int(args[3])
    # By how much % to increase the available set size
    percentage_increase = np.int(args[4])
    # How precise oracle is in %
    oracle_correct = np.int(args[5])
    # Method to choose sample to be labeled
    mthd = args[2]
    # Ouput file path
    out = args[6]

    return dataset, mdl, mthd, percentage_limit, percentage_increase, oracle_correct, out


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


# Compute error rate manually, for graph projection
def getError(Model, Data, Labels):
    classProb = Model.predict(x=Data, verbose=1)
    correctProb = (classProb * Labels).sum(axis=1)
    wrongProb = (classProb * (1-Labels)).max(axis=1)

    accuracy = (correctProb > wrongProb).mean()
    error = 1.0 - accuracy
    print("\n acc: \n", accuracy)
    return error


# Simulate faulty oracle that doesn't always get the label right
def oracleSim(oracle_correct, labels):
    edited_labels = []
    # Get the number of classes
    class_number = int(labels.shape[1])

    # Iterate throught labels to be added
    for l in labels:
        addition = [l]
        chance = randint(1,100)

        # Given the oracle precision chance, botch the known label
        if chance < oracle_correct:
            # Possible new label 0 --- class number minus 1
            botched_label = randint(0, class_number - 1)
            addition = mapLabelsOneHot(botched_label)

        # Append new/old label
        edited_labels = np.append(edited_labels, addition, axis=0)

    return edited_labels


# Active learning loop
def actLearning(model, percentage_limit, oracle_correct, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels):
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

        if oracle_correct != 100:
            additionL = oracleSim(oracle_correct, additionL)
        #additionL = np.expand_dims(additionL, axis=0)

        actTrnData = np.append(actTrnData, addition, axis=0)
        actTrnLabels = np.append(actTrnLabels, additionL, axis=0)
        print("", actTrnData.shape)

        # Add high certainty samples temporarily TODOOOOOOOOOO
        # Train on the new training batch
        model.fit(
            x=actTrnData, y=actTrnLabels,
            batch_size=48, epochs=1, verbose=1,
            validation_data=[tstData, tstLabels], shuffle=True)
        # Keep error/labeled %
        err = np.append(err, getError(model, tstData, tstLabels))
        labeled = np.append(labeled, 100*(float(len(actTrnData))/float(len(trnData))))

    print("",err)
    print("",labeled)
    return err, labeled


# Passive learning loop (for baseline)
def passLearning(model, percentage_limit, percentage_increase, trnData, trnLabels, tstData, tstLabels):
    # Initial error
    err = getError(model, tstData, tstLabels)
    # Initial labeled %
    labeled = [10]

    # Loop to make it comparable to active learning (runs for the same number of epochs)
    while max(labeled) < percentage_limit:

        # Train on the new training batch
        model.fit(
            x=trnData, y=trnLabels,
            batch_size=48, epochs=1, verbose=1,
            validation_data=[tstData, tstLabels], shuffle=True)
        # Keep error/labeled %
        err = np.append(err, getError(model, tstData, tstLabels))
        labeled.append(max(labeled) + percentage_increase)
        print(labeled)

    print("",err)
    print("",labeled)
    return err, labeled


def main():
    # Get model, method and number of querries
    dataset, mdl, mthd, percentage_limit, percentage_increase, oracle_correct, outfile = getArgs()
    # Set up tensorflow
    tfSetUp()
    # Read dataset
    trnData, tstData, trnLabels, tstLabels, actTrnData, actTrnLabels, actPoolData, actPoolLabels, class_number = getData(dataset)

    # Number of samples to label in one go (1%)
    K = len(trnData)/100 * percentage_increase

    model = getModel(mdl, actTrnData, actTrnLabels, tstData, tstLabels, class_number)
    # Save weights after model creation and slight training
    model.save_weights('tmpweights.h5')

    # Error definition for methods
    baseErr = 0
    rsErr = 0
    lcErr = 0
    smErr = 0
    entErr = 0
    labeled = 0

    # Run active learning with only one method or compare several
    if mthd == "base" or mthd == "all":
        baseErr, labeled = passLearning(model, percentage_limit, percentage_increase, rnData, trnLabels, tstData, tstLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "rs" or mthd == "all":
        qs = RandomSampling(model, actPoolData)
        rsErr, labeled = actLearning(model, percentage_limit, oracle_correct, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "lc" or mthd == "all":
        qs = UncertaintySampling(model, "lc", actPoolData)
        lcErr, labeled = actLearning(model, percentage_limit, oracle_correct, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "sm" or mthd == "all":
        qs = UncertaintySampling(model, "sm", actPoolData)
        smErr, labeled = actLearning(model, percentage_limit, oracle_correct, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "ent" or mthd == "all":
        qs = UncertaintySampling(model, "ent", actPoolData)
        entErr, labeled = actLearning(model, percentage_limit, oracle_correct, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    # Save data to a npz file
    np.savez(outfile, labeled=labeled, base=baseErr, rs=rsErr, lc=lcErr, sm=smErr, ent=entErr, dset=dataset, model=mdl, method=mthd)

if __name__ == '__main__':
    main()
