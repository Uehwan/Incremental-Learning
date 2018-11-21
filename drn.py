"""
    [drn.py]
    - Author: Uehwan, Kim
    - Affiliation: Korea Advanced Institute of Science and Technology (KAIST)
    - E-mail: ohnefetter@kaist.ac.kr
    - Repository URL: https://github.com/Uehwan
    - Licence:
    - Content
    1) Developmental Resonance Network (DRN)
        > G.-M. Park, J.-W. Choi and J.-H. Kim, "Developmental Resonance Network,"
          IEEE Trans. on Neural Networks and Learning Systems, Aug. 2018.
        > Implementation of DRN network and test cases
        > Original DRN is based on Fusion ART, but current DRN is based on Fuzzy ART.
    2) DRNMAP
        > U.-H. Kim and J.-H. Kim, "DRNMAP: ~~,"
          IEEE Trans. on Neural Networks and Learning Systems, Dec. 2018, Submitted.
        > Implementation of DRNMAP algorithm and performance analysis
        > DRNMAP employs rDRN rather than the original DRN
        > rDRN stands for reduced-DRN, which employs simplified grouping
    - Development history
        > 18.11.09.: Git configuration & initial file generation
        > 18.11.10.: DRN implementation 1) init method
        > 18.11.13.: DRN implementation 1) weight updates, 2) activation
        > 18.11.14.: DRN implementation 1) train, 2) template matching
        > 18.11.16.: DRN implementation 1) group-related functions
        > 18.11.20.: DRN testing & debugging
        > 18.11.21.: DRNMAP implementation 1) rDRN
    - TO DO's
        > Extend Fuzzy DRN => Fusion DRN 
"""

import numpy as np
from functools import partial
import scipy.io as io

l2_norm = partial(np.linalg.norm, ord=2, axis=-1)


class DRN(object):
    def __init__(self, lr=1.0, glr=1.0, alpha=1.0, rho=0.9, v=2):

        self.lr = lr          # learning rate
        self.glr = glr        # global learning rate
        self.alpha = alpha    # parameter for node activation
        self.rho = rho        # vigilance parameter
        self.v = v            # number of nodes to select

        self.w = None         # weights for clusters; (samples, weights)
        self.wg = None        # global weight vector
        self.dim = None       # dimension of input vectors
        self.n_category = 0   # number of categories
        self.group = []       # group container

    def _init_weights(self, sample):
        assert len(sample.shape) == 1, "Wrong vector for initialization, " \
                                       "just one feature vector is required"
        self.w, self.wg = [np.atleast_2d(np.hstack((sample, sample)))] * 2
        self.dim = sample.shape[0]
        self.n_category += 1
        print("0. Weight Initialization")

    def _update_global_weight(self, sample):
        if self._distance(sample, self.wg):
            print("7. GLOBAL UPDATE CALLED")
            self.wg = self._update_weight(sample, self.wg, self.glr)
            print("7-1. PRINTING SELF GROUP BEFORE")
            print(self.group)
            self._grouping()
            print("7-1. PRINTING SELF GROUP AFTER")
            print(self.group)

    def _update_weight(self, sample, weight, lr):
        print("6. UPDATE WEIGHT CALLED")
        _, _, w1, w2 = self._split_weight(sample, weight)
        updated_weight = lr * np.hstack((w1, w2)) + (1 - lr) * weight
        return updated_weight

    def _split_weight(self, sample, weight):
        print("5. SPLIT WEIGHT CALLED")
        weight = np.atleast_2d(weight)
        front, back = weight[:, :self.dim], weight[:, self.dim:]
        print("shapes of splited vectors")
        print(front.shape, back.shape, sample.shape)
        w1, w2 = np.minimum(sample, front), np.maximum(sample, back)
        return front, back, w1, w2

    def _grouping(self):
        print("10. GROUPING CALLED")
        to_delete = []
        for ind, connection in enumerate(self.group):
            smaller, larger = connection[:2]  # comparison in the index order not the size
            resonance, w_ij_1, w_ij_2 = self._resonance_between_clusters(self.w[smaller], self.w[larger])
            if resonance:
                # merge two clusters
                self.w[smaller] = np.hstack((w_ij_1, w_ij_2))
                self.w = np.delete(self.w, larger, axis=0)
                self.n_category -= 1
                # reconnect nodes previously connected to "larger" to "smaller"
                for check in self.group:
                    if larger in check[:2]:
                        check[check.index(larger)] = smaller
                        item1, item2 = min(check[0], check[1]), max(check[0], check[1])
                        if item1 == item2:
                            T = 0
                            to_delete.append([item1, item2, 0])
                        else:
                            print("printing items")
                            print("SELF W", self.w, self.n_category)
                            print(item1, item2)
                            subtraction = np.atleast_2d(self.w[item1] - self.w[item2])
                            center_of_mass_diff = (subtraction[:, :self.dim] + subtraction[:, self.dim:]) / 2
                            T = np.exp(-self.alpha * l2_norm(center_of_mass_diff))
                        check[0], check[1], check[2] = item1, item2, T
        # remove inappropriate connections
        [self.group.remove(delete) for delete in to_delete]

    def _resonance_between_clusters(self, cluster1, cluster2):
        front, back = self.wg[:, :self.dim], self.wg[:, self.dim:]
        M = np.prod(back - front)
        print("4. RESONANCE BETWEEN CLUSTERS")
        cluster1, cluster2 = np.atleast_2d(cluster1), np.atleast_2d(cluster2)
        w_i_front, w_i_back = cluster1[:, :self.dim], cluster1[:, self.dim:]
        w_j_front, w_j_back = cluster2[:, :self.dim], cluster2[:, self.dim:]
        w_ij_front, w_ij_back = np.minimum(w_i_front, w_j_front), np.maximum(w_i_back, w_j_back)
        # _, _, w_ij_front, w_ij_back = self._split_weight(cluster1, cluster2)
        S = np.prod(w_ij_back - w_ij_front)
        return (M - S) / M > self.rho, w_ij_front, w_ij_back

    def _distance(self, sample, weight):
        print("3. DISTANCE FUNC CALLED")
        f, b, w1, w2 = self._split_weight(sample, weight)
        distance = l2_norm(w1 - f + w2 - b)
        return distance

    def _activation(self, sample):
        print("8. ACTIVATION CALLED")
        distance = self._distance(sample, self.w)
        activation = np.exp(-self.alpha * distance)
        return activation

    def _template_matching(self, sample, v_nodes):
        print("1. TEMPLATE MATCHING CALLED")
        front, back, _, _ = self._split_weight(sample, self.wg)
        M = np.prod(back - front)
        print("2. TEMPLATE MATCHING SECOND WEIGHT SPLIT")
        _, _, w1, w2 = self._split_weight(sample, self.w[v_nodes[0]])
        S = np.prod(w2 - w1)
        return (M - S) / M > self.rho

    def _add_category(self, sample):
        print("12. ADD CATEGORY")
        self.n_category += 1
        new_weight = np.hstack((sample, sample))
        self.w = np.vstack((self.w, new_weight))

    def _add_group(self, v_nodes, sample, condition):
        print("11. ADD GROUP")
        print("V_NODES 11: ", v_nodes)
        front, back, _, _ = self._split_weight(sample, self.w[v_nodes])
        center_of_mass = (front + back) / 2
        to_connect = np.copy(v_nodes)
        print("V_NODES 22: ", to_connect)
        print("n_category: ", self.n_category)
        if not condition:
            center_of_mass = np.vstack((center_of_mass, sample))
            to_connect = np.hstack((to_connect, self.n_category - 1))

        print("V_NODES 33: ", to_connect)
        for first in range(len(to_connect)):
            for second in range(first, len(to_connect)):
                smaller, larger = sorted([to_connect[first], to_connect[second]])
                # new connections get added (first condition)
                # and synaptic strengths get updated (second condition)
                if not smaller == larger and not (smaller, larger) in self.group or condition and v_nodes[0] in (smaller, larger):
                    T = np.exp(-self.alpha * l2_norm(center_of_mass[first] - center_of_mass[second]))
                    self.group.append([smaller, larger, T])

    def train(self, x, epochs=1):
        """
        Description
            - Train DRN model with the input vectors
            - DRN automatically clusters the input vectors

        Parameters
            - x: 2d array of size (samples, features), where all features can
                 range [-inf, inf]
            - epochs: the number of training loop

        Return
            - self: DRN class itself
        """
        # check if the input vectors (x) are of shape (samples, features)
        assert len(x.shape) == 2, "Wrong input vector shape: input.shape = (samples, features)"

        # training for "epochs" times
        for epoch in range(epochs):
            # randomly shuffle data
            i = 0
            for sample in np.random.permutation(x):
                print("printing i: " + str(i))
                # init the cluster weights for the first input vector
                if self.w is None and self.wg is None:
                    self._init_weights(x[0])
                    continue

                # global vector update
                self._update_global_weight(sample)

                # node activation & template matching
                activations = self._activation(sample)
                v_node_selection = np.argsort(activations)[::-1][:self.v]

                # check if resonance occurred
                resonance = self._template_matching(sample, v_node_selection)
                if resonance:
                    # update weight for the cluster
                    category = v_node_selection[0]
                    self.w[category] = self._update_weight(sample, self.w[category], self.lr)
                else:
                    # no matching occurred
                    print("NEW CATEGORY")
                    print("NEW CATEGORY")
                    self._add_category(sample)

                # group the v-nodes
                if self.n_category > 1:
                    self._add_group(v_node_selection, sample, resonance)
                i += 1
        return self

    def test(self, x, train=False):
        """
        Description
            - Test the trained DRN model with the input vectors
            - DRN predicts the clusters of the input vectors

        Parameters
            - x: 2d array of size (samples, features), where all features can
                 range [-inf, inf]
            - train: to train the DRN model with the input vectors for test

        Return
            - self: DRN class itself
        """
        pass


class rDRN(DRN):
    def _grouping(self):
        pass

    def _add_group(self):
        pass


class DRNMAP(object):
    def __init__(self, lr=0.9, glr=1.0, alpha=1.0, rho=0.9, v=2):
        self.drn = DRN(lr, glr, alpha, rho, v)


if __name__ == '__main__':
    import random

    random.seed(7810)
    print("TEST of DRN")
    # data = io.loadmat('data.mat')['points']
    data_test = np.array([[1, 1],
                          [1.5, 0.7],
                          [0.8, 1.2],
                          [-2, 1.3],
                          [-1.5, -0.6],
                          [3, -1],
                          [-2.4, -1.5],
                          [-2.7, 3],
                          [2.9, -1.2],
                          [10, -10],
                          [-10, 10],
                          [5, 8]])
    drn = DRN()
    # sample1 = data[45]
    # dummy_w = np.hstack((data[:10], data[30:40]))
    # ff, bb = dummy_w[:, :drn.dim], dummy_w[:, drn.dim:]
    # ww1, ww2 = np.minimum(sample1, ff), np.maximum(sample1, bb)
    # drn.train(data)
    # print(drn.n_category)
