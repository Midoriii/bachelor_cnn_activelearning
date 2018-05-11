import numpy as np
from tools.tools import mapLabelsOneHot


# Method for creating and adding pseudolabels where the model is confident
def pseudoLabel(model, labeled, actTrnData, actPoolData, actPoolLabels):
    # Get the probabilities
    probs = model.predict(x=actPoolData, verbose=1)
    # Using entropy to determine trustworthy samples
    ent_score = np.sum(-probs * np.log(probs), axis=1)
    # Threshold for selecting highly certain samples and its decay
    threshold_base = 0.005
    threshold_decay = 0.000033
    # -9 because labeled % starts at 10
    threshold = threshold_base - ((max(labeled) - 9) * threshold_decay)

    # Get indices of images/labels, that show low entropy
    indices = np.where(ent_score[:] <= threshold)
    deletion_indices = []
    indices = indices[0]


    # Gotta add check to not add images already in actTrnData
    # For the indices of promising pics in actPoolData
    for i in indices:
        # For the images in training set already
        for j in actTrnData:
            # Is the promising pic already there ?
            if (actPoolData[i] == j).all():
                deletion_indices.append(i)
                break

    if len(deletion_indices) > 0:
        indices = np.delete(indices, deletion_indices)

    # Data and labels to temporarily add to training set
    data_addition = actPoolData[indices]
    label_addition = np.array([]).reshape(0, int(actPoolLabels.shape[1]))

    for i in indices:
        wishful_label = np.argmax(probs[i,:])
        #print(wishful_label)
        label_addition = np.append(label_addition, mapLabelsOneHot([wishful_label], 'yes', int(actPoolLabels.shape[1])), axis=0)

    print(len(data_addition))

    return data_addition, label_addition
