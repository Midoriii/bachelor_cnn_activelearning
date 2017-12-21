from __future__ import print_function
import sys
import os.path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

# Cesta k vysledkum/datum
i = 1
data_path = "results/exp" + str(i) + ".npz"

lcErr = 0
smErr = 0
entErr = 0
labeled = 0
dataset = ''
method = ''
model = ''

# For every file containing data
while os.path.isfile(data_path):
  
    # Loading file containing data
    graph_data = np.load(data_path)

    # Load data into variables for plotting
    lcErr = graph_data['lc']
    smErr = graph_data['sm']
    entErr = graph_data['ent']
    labeled = graph_data['labeled']
    dataset = graph_data['dset']
    method = graph_data['method']
    model = graph_data['model']

    # Plot data based on method used
    if method == 'lc' or method == 'all':
        plt.plot(labeled, lcErr, 'g', label='LC')
    if method == 'sm' or method == 'all':
        plt.plot(labeled, smErr, 'r', label='SM')
    if method == 'ent' or method == 'all':
        plt.plot(labeled, entErr, 'b', label='ENT')

    # Add labels and title
    plt.xlabel('Labeled %')
    plt.ylabel('Error')
    plt.title(str(dataset) + "_" + str(model) + "_" + str(method))
    axes = plt.gca()
    axes.set_xlim(10, max(labeled))
    axes.set_ylim(0.0, 1.0)

    # Add legend to the graph
    legend = plt.legend(loc='upper right')
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    
    for label in legend.get_texts():
        label.set_fontsize('medium')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    figpath = "results/exp" + str(i) + ".png"

    # Plot graphs to a file
    plt.savefig(figpath)
    # Clear figure
    plt.clf()

    i += 1
    data_path = "results/exp" + str(i) + ".npz"
