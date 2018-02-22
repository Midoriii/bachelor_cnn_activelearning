from __future__ import print_function
import sys
import os.path

from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

# Cesta k vysledkum/datum
i = 1
data_path = "results/uncertainty/exp" + str(i) + ".npz"

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
    baseErr = graph_data['base']
    baseErr[:] = [x * 100.0 for x in baseErr]
    rsErr = graph_data['rs']
    rsErr[:] = [x * 100.0 for x in rsErr]
    lcErr = graph_data['lc']
    lcErr[:] = [x * 100.0 for x in lcErr]
    smErr = graph_data['sm']
    smErr[:] = [x * 100.0 for x in smErr]
    entErr = graph_data['ent']
    entErr[:] = [x * 100.0 for x in entErr]
    labeled = graph_data['labeled']
    dataset = graph_data['dset']
    method = graph_data['method']
    model = graph_data['model']

    baseErr = savgol_filter(baseErr, 27, 2)
    rsErr = savgol_filter(rsErr, 27, 2)
    lcErr = savgol_filter(lcErr, 27, 2)
    smErr = savgol_filter(smErr, 27, 2)
    entErr = savgol_filter(entErr, 27, 2)

    # Plot data based on method used
    if method == 'base' or method == 'all':
        plt.plot(labeled, baseErr, linestyle = ':', color='C1', label='Base')
    if method == 'rs' or method == 'all':
        plt.plot(labeled, rsErr, linestyle=':', color='y', label='RS')
    if method == 'lc' or method == 'all':
        plt.plot(labeled, lcErr, 'g', label='LC')
    if method == 'sm' or method == 'all':
        plt.plot(labeled, smErr, 'r', label='SM')
    if method == 'ent' or method == 'all':
        plt.plot(labeled, entErr, 'b', label='ENT')

    # Add labels and title
    plt.xlabel('Labeled %')
    plt.ylabel('Error %')
    plt.yscale('log')
    plt.title(str(dataset) + "_" + str(model) + "_" + str(method))
    axes = plt.gca()
    axes.set_xlim(10, max(labeled))
    axes.set_ylim(5, 10**2)
    axes.grid(which='both', axis='y', linestyle='--')

    # Add legend to the graph
    legend = plt.legend(loc='upper right')
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    for label in legend.get_texts():
        label.set_fontsize('medium')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    figpath = "results/uncertainty/exp" + str(i) + ".png"

    # Plot graphs to a file
    plt.savefig(figpath)
    # Clear figure
    plt.clf()

    i += 1
    data_path = "results/uncertainty/exp" + str(i) + ".npz"
