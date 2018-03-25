from __future__ import print_function
import sys
import os.path

from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np


# Get arguments from console
args = sys.argv[1:]
exp_path = args[0]

# Cesta k vysledkum/datum
i = 1
data_path = "results/" + exp_path + str(i) + ".npz"

lcErr = 0
smErr = 0
entErr = 0
rsErr = 0
kcgErr = 0
labeled = 0
baseErr = 0
dataset = ''
method = ''
model = ''

print("Plotting " + exp_path)

# For every file containing data
while os.path.isfile(data_path):

    # Loading file containing data
    graph_data = np.load(data_path)

    # Load data into variables for plotting
    baseErr = graph_data['base']
    baseErr[:] = [x * 100.0 for x in baseErr]
    rsErr = graph_data['rs']
    rsErr[:] = [x * 100.0 for x in rsErr]
    kcgErr = graph_data['kcg']
    kcgErr[:] = [x * 100.0 for x in kcgErr]
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

    # Make single line from baseline error -- only the best score, the unreachable border
    baseErrMax = min(baseErr)
    baseErrLine = np.empty(len(labeled))
    baseErrLine.fill(baseErrMax)

    rsErr = savgol_filter(rsErr, 15, 9)
    lcErr = savgol_filter(lcErr, 15, 9)
    smErr = savgol_filter(smErr, 15, 9)
    entErr = savgol_filter(entErr, 15, 9)
    kcgErr = savgol_filter(kcgErr, 15, 9)

    # Plot data based on method used
    if method == 'base' or method == 'all':
        plt.plot(labeled, baseErrLine, linestyle = ':', color='C1', label='Base')
    if method == 'rs' or method == 'all':
        plt.plot(labeled, rsErr, linestyle=':', color='y', label='RS')
    if method == 'lc' or method == 'all':
        plt.plot(labeled, lcErr, 'g', label='LC')
    if method == 'sm' or method == 'all':
        plt.plot(labeled, smErr, 'r', label='SM')
    if method == 'ent' or method == 'all':
        plt.plot(labeled, entErr, 'b', label='ENT')
    if method == 'kcg' or method == 'all':
        plt.plot(labeled, kcgErr, 'k', label='KCG')

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

    figpath = "results/" + exp_path + str(i) + ".png"

    # Plot graphs to a file
    plt.savefig(figpath)
    # Clear figure
    plt.clf()

    i += 1
    data_path = "results/" + exp_path + str(i) + ".npz"
