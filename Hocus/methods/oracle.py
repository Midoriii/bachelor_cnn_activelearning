import numpy as np
from random import randint
from tools.tools import mapLabelsOneHot


# Simulate faulty oracle that doesn't always get the label right
def oracleSim(oracle_correct, labels):
    # Get the number of classes
    class_number = int(labels.shape[1])
    edited_labels = np.array([]).reshape(0, class_number)

    # Iterate throught labels to be added
    for l in labels:
        addition = l
        addition = np.expand_dims(addition, axis=0)
        chance = randint(1,100)

        # Given the oracle precision chance, botch the known label
        if chance > oracle_correct:
            # Possible new label 0 --- class number minus 1
            botched_label = [randint(0, class_number - 1)]
            addition = mapLabelsOneHot(botched_label, 'yes', class_number)

        # Append new/old label
        #print(addition)
        #print(edited_labels)
        edited_labels = np.append(edited_labels, addition, axis=0)

    return edited_labels
