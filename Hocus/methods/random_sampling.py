import numpy as np
from random import randint

class RandomSampling():
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset


    def querry(self, K):
        ask_id = []
        # We want a K number of unique random additions
        while len(set(ask_id)) < K:
            # Get a random int from 0:dataset length
            random_addition = randint(0, self.dataset.shape[0] - 1)
            # Append to indices
            ask_id = np.append(ask_id, random_addition)

        # Choose only unique indices and map to int
        ask_id = list(set(ask_id))
        ask_id = map(int, ask_id)

        return ask_id
