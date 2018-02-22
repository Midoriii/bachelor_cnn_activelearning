import numpy as np

class UncertaintySampling():
    def __init__(self, model, method, dataset):
        self.model = model
        self.dataset = dataset
        self.method = method


    def querry(self, K):
        classProb = self.model.predict(x=self.dataset)

        if self.method == 'lc':
            score = -np.max(classProb, axis=1)

        elif self.method == 'sm':
            if np.shape(classProb)[1] > 2:
                # Find 2 largest class probabilities
                classProb = -(np.partition(-classProb, 2, axis=1)[:, :2])
            score = -np.abs(classProb[:, 0] - classProb[:, 1])

        elif self.method == 'ent':
            score = np.sum(-classProb * np.log(classProb), axis=1)

        # Get the best K indices
        ask_id = np.argpartition(score, -K)[-K:]

        return ask_id
