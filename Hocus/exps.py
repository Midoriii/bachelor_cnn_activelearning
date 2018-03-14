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
from methods.k_cent_greedy import KCenterGreedy
from methods.oracle import oracleSim
from methods.pseudo_labeling import pseudoLabel

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
    # Method to choose sample to be labeled
    mthd = args[2]
    # What amount of dataset to label
    percentage_limit = np.int(args[3])
    # By how much % to increase the available set size
    percentage_increase = np.int(args[4])
    # How precise oracle is in %
    oracle_correct = np.int(args[5])
    # Use pseudolabels?
    pseudo_labels = args[6]
    # Ouput file path
    out = args[7]

    return dataset, mdl, mthd, percentage_limit, percentage_increase, oracle_correct, pseudo_labels, out


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


# Active learning loop
def actLearning(model, percentage_limit, oracle_correct, pseudo_labels, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels):
    # Initial error
    err = getError(model, tstData, tstLabels)
    # Initial labeled %
    labeled = [int(100*(float(len(actTrnData))/float(len(trnData))))]

    already_asked_ids = []

    # While actively labeled percentage is lesser than the desired
    while max(labeled) < percentage_limit:

        # Add uncertain sample to actTrnData and its label to actTrnLabels
        ask_id = qs.querry(K, already_asked_ids)
        already_asked_ids.extend(ask_id)

        addition = actPoolData[ask_id]
        #addition = np.expand_dims(addition, axis=0)
        additionL = actPoolLabels[ask_id]

        if oracle_correct != 100:
            additionL = oracleSim(oracle_correct, additionL)
        #additionL = np.expand_dims(additionL, axis=0)

        actTrnData = np.append(actTrnData, addition, axis=0)
        actTrnLabels = np.append(actTrnLabels, additionL, axis=0)

        # Add high certainty samples temporarily TODOOOOOOOOOO
        if(pseudo_labels == 'yes'):
            pseudoLabeledActTrnData = actTrnData
            pseudoLabeledActTrnLabels = actTrnLabels

            pseudolabeled_addition, pseudolabeled_additionL = pseudoLabel(model, labeled, actTrnData, actPoolData, actPoolLabels)

            pseudoLabeledActTrnData = np.append(pseudoLabeledActTrnData, pseudolabeled_addition, axis=0)
            pseudoLabeledActTrnLabels = np.append(pseudoLabeledActTrnLabels, pseudolabeled_additionL, axis=0)


            # Train on the new training batch with temporary pseudolabels
            model.fit(
                x=pseudoLabeledActTrnData, y=pseudoLabeledActTrnLabels,
                batch_size=48, epochs=1, verbose=1,
                validation_data=[tstData, tstLabels], shuffle=True)
        else:
            # Train on the new training batch
            model.fit(
                x=actTrnData, y=actTrnLabels,
                batch_size=48, epochs=1, verbose=1,
                validation_data=[tstData, tstLabels], shuffle=True)
        # Keep error/labeled %
        err = np.append(err, getError(model, tstData, tstLabels))
        labeled = np.append(labeled, int(100*(float(len(actTrnData))/float(len(trnData)))))

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
        print("Getting error")
        err = np.append(err, getError(model, tstData, tstLabels))
        labeled.append(max(labeled) + percentage_increase)
        print(labeled)

    print("",err)
    print("",labeled)
    return err, labeled


def main():
    # Get model, method and number of querries
    dataset, mdl, mthd, percentage_limit, percentage_increase, oracle_correct, pseudo_labels, outfile = getArgs()
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
    kcgErr = 0
    lcErr = 0
    smErr = 0
    entErr = 0
    labeled = 0

    # Run active learning with only one method or compare several
    if mthd == "base" or mthd == "all":
        print("\nRunning base\n")
        baseErr, labeled = passLearning(model, percentage_limit, percentage_increase, trnData, trnLabels, tstData, tstLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "rs" or mthd == "all":
        print("\nRunning rs\n")
        qs = RandomSampling(model, actPoolData)
        rsErr, labeled = actLearning(model, percentage_limit, oracle_correct, pseudo_labels, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "lc" or mthd == "all":
        print("\nRunning lc\n")
        qs = UncertaintySampling(model, "lc", actPoolData)
        lcErr, labeled = actLearning(model, percentage_limit, oracle_correct, pseudo_labels, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "sm" or mthd == "all":
        print("\nRunning sm\n")
        qs = UncertaintySampling(model, "sm", actPoolData)
        smErr, labeled = actLearning(model, percentage_limit, oracle_correct, pseudo_labels, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "ent" or mthd == "all":
        print("\nRunning ent\n")
        qs = UncertaintySampling(model, "ent", actPoolData)
        entErr, labeled = actLearning(model, percentage_limit, oracle_correct, pseudo_labels, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    if mthd == "kcg" or mthd == "all":
        print("\nRunning kcg\n")
        qs = KCenterGreedy(actPoolData)
        kcgErr, labeled = actLearning(model, percentage_limit, oracle_correct, pseudo_labels, qs, K, actTrnData, trnData, actTrnLabels, tstData, tstLabels, actPoolData, actPoolLabels)
        # Load inital weights for other methods to use
        model.load_weights('tmpweights.h5')

    # Save data to a npz file
    np.savez(outfile, labeled=labeled, base=baseErr, rs=rsErr, kcg=kcgErr, lc=lcErr, sm=smErr, ent=entErr, dset=dataset, oc=oracle_correct, pseudol=pseudo_labels, model=mdl, method=mthd)

if __name__ == '__main__':
    main()
