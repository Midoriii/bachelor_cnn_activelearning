# https://arxiv.org/pdf/1708.00489.pdf
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist


class KCenterGreedy():
    def __init__(self, x):
        self.x = x
        self.flat_x = self.flatten()
        self.features = self.flat_x
        self.min_distances = None
        self.n_obs = self.x.shape[0]
        self.already_selected = []


    def flatten(self):
        shape = self.x.shape
        flat_x = self.x

        if len(shape) > 2:
          flat_x = np.reshape(self.x, (shape[0],np.product(shape[1:])))

        return flat_x


    def update_distances(self, cluster_centers):

        cluster_centers = [d for d in cluster_centers
                             if d not in self.already_selected]

        if cluster_centers:
          # Update min_distances for all examples given new cluster center.
          x = self.features[cluster_centers]
          dist = cdist(self.features, x, 'euclidean')

          if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1,1)
          else:
            self.min_distances = np.minimum(self.min_distances, dist)


    def querry(self, K, already_asked_ids):

        self.update_distances(already_asked_ids)
        new_batch = []

        for _ in range(K):
          if self.already_selected is None:
            # Initialize centers
            ind = np.random.choice(np.arange(self.n_obs))
          else:
            ind = np.argmax(self.min_distances)

          # Shouldn't be a cluster center
          assert ind not in already_asked_ids

          self.update_distances([ind])

          new_batch.append(ind)


        self.already_selected = already_asked_ids

        return new_batch
