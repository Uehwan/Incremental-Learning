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
        > 18.11.28.: DRN testing & debugging (compare against MATLAB DRN)
    - TO DO's
        > Extend Fuzzy DRN => Fusion DRN 
"""

import numpy as np
from functools import partial
import scipy.io as io

l2_norm = partial(np.linalg.norm, ord=2, axis=-1)


class DRN(object):
    def __init__(self, lr=0.9, glr=1.0, alpha=1.0, rho=0.9, v=2):

        self.lr = lr          # learning rate
        self.glr = glr        # global learning rate
        self.alpha = alpha    # parameter for node activation
        self.rho = rho        # vigilance parameter
        self.v = v            # number of nodes to select

        self.w = None         # weights for clusters; (samples, weights)
        self.wg = None        # global weight vector
        self.dim = None       # dimension of input vectors
        self.n_category = 0   # number of categories
        self.group = {}       # group container

    def _init_weights(self, sample):
        assert len(sample.shape) == 1, "Wrong vector for initialization, " \
                                       "just one feature vector is required"
        self.w, self.wg = [np.atleast_2d(np.hstack((sample, sample)))] * 2
        self.dim = sample.shape[0]
        self.n_category += 1

    def _update_global_weight(self, sample):
        if self._distance(sample, self.wg):
            self.wg = self._update_weight(sample, self.wg, self.glr)
            if len(self.group) > 0:
                self._grouping()

    def _update_weight(self, sample, weight, lr):
        _, _, w1, w2 = self._split_weight(sample, weight)
        updated_weight = lr * np.hstack((w1, w2)) + (1 - lr) * weight
        return updated_weight

    def _split_weight(self, sample, weight):
        weight = np.atleast_2d(weight)
        front, back = weight[:, :self.dim], weight[:, self.dim:]
        w1, w2 = np.minimum(sample, front), np.maximum(sample, back)
        return front, back, w1, w2

    def _grouping(self):
        while True:
            resonance, idx_s, idx_l, w_ij_1, w_ij_2 = self._condition_for_grouping()
            if not resonance:
                break

            # merge two clusters
            self.w[idx_s] = np.hstack((w_ij_1, w_ij_2))
            to_delete_group = []

            # reconnect nodes previously connected to "idx_l" to "idx_s"
            for check in self.group:
                if idx_l in check:
                    item1, item2 = check
                    reconnection = {item1, item2}
                    _, _ = reconnection.remove(idx_l), reconnection.add(idx_s)
                    reconnection = sorted(reconnection)
                    if not len(reconnection) == 2:  # self-connection
                        to_delete_group.append((item1, item2))
                    else:  # update the synaptic strength
                        item1, item2 = reconnection
                        subtraction = np.atleast_2d(self.w[item1] - self.w[item2])
                        center_of_mass_diff = (subtraction[:, :self.dim] + subtraction[:, self.dim:]) / 2
                        T = np.exp(-self.alpha * l2_norm(center_of_mass_diff))
                        self.group[(item1, item2)] = T

            # delete the collected items from group and update w and n_category
            for delete in to_delete_group:
                del self.group[delete]
            self.w = np.delete(self.w, idx_l, axis=0)
            self.n_category -= 1

            # update indices > idx_l
            to_delete_group.clear()
            to_add_group = []
            for check in self.group:
                if any([c > idx_l for c in check]):
                    item1, item2 = check
                    residuals = [-1 if item > idx_l else 0 for item in check]
                    item1, item2 = item1 - residuals[0], item2 - residuals[1]
                    to_add_group.append([(item1, item2), self.group[check]])
                    # self.group[(item1, item2)] = self.group[check]
                    to_delete_group.append(check)

            for pair, strength in to_add_group:
                self.group[pair] = strength

            for delete in to_delete_group:
                del self.group[delete]

    def _condition_for_grouping(self):
        for idx_s, idx_l in self.group:
            resonance, w_ij_1, w_ij_2 = self._resonance_between_clusters(idx_s, idx_l)
            if resonance:
                return resonance, idx_s, idx_l, w_ij_1, w_ij_2
        return False, idx_s, idx_l, w_ij_1, w_ij_2
        # idx_s, idx_l = max(self.group, key=lambda x: self.group[x])
        # return resonance, idx_s, idx_l, w_ij_1, w_ij_2

    def _resonance_between_clusters(self, idx_s, idx_l):
        front, back = self.wg[:, :self.dim], self.wg[:, self.dim:]
        M = np.prod(back - front)
        cluster1, cluster2 = np.atleast_2d(self.w[idx_s]), np.atleast_2d(self.w[idx_l])
        w_i_front, w_i_back = cluster1[:, :self.dim], cluster1[:, self.dim:]
        w_j_front, w_j_back = cluster2[:, :self.dim], cluster2[:, self.dim:]
        w_ij_front, w_ij_back = np.minimum(w_i_front, w_j_front), np.maximum(w_i_back, w_j_back)
        S = np.prod(w_ij_back - w_ij_front)
        return (M - S) / M > self.rho, w_ij_front, w_ij_back

    def _distance(self, sample, weight):
        f, b, w1, w2 = self._split_weight(sample, weight)
        distance = l2_norm(w1 - f + w2 - b)
        return distance

    def _activation(self, sample):
        distance = self._distance(sample, self.w)
        activation = np.exp(-self.alpha * distance)
        return activation

    def _template_matching(self, sample, v_nodes):
        front, back, _, _ = self._split_weight(sample, self.wg)
        M = np.prod(back - front)
        _, _, w1, w2 = self._split_weight(sample, self.w[v_nodes[0]])
        S = np.prod(w2 - w1)
        return (M - S) / M > self.rho

    def _add_category(self, sample):
        self.n_category += 1
        new_weight = np.hstack((sample, sample))
        self.w = np.vstack((self.w, new_weight))

    def _add_group(self, v_nodes, sample, condition):
        front, back, _, _ = self._split_weight(sample, self.w[v_nodes])
        center_of_mass = (front + back) / 2
        to_connect = np.copy(v_nodes)

        if not condition:
            center_of_mass = np.vstack((center_of_mass, sample))
            to_connect = np.hstack((to_connect, self.n_category - 1))

        for first in range(len(to_connect)):
            for second in range(first + 1, len(to_connect)):
                smaller, larger = sorted([to_connect[first], to_connect[second]])
                # new connections get added (first condition)
                # and synaptic strengths get updated (second condition)
                T = np.exp(-self.alpha * l2_norm(center_of_mass[first] - center_of_mass[second]))
                if not T == 0 or v_nodes[0] in (smaller, larger):
                    self.group[(smaller, larger)] = T

    def train(self, x, epochs=1, shuffle=True, train=True):
        """
        Description
            - Train DRN model with the input vectors
            - DRN automatically clusters the input vectors

        Parameters
            - x: 2d array of size (samples, features), where all features can
                 range [-inf, inf]
            - epochs: the number of training loop
            - permute: whether to shuffle the x's when training

        Return
            - self: DRN class itself
            - classes: the categories of each sample
        """
        # check if the input vectors (x) are of shape (samples, features)
        x = np.atleast_2d(x)
        assert len(x.shape) == 2, "Wrong input vector shape: input.shape = (samples, features)"

        classes = []
        # training for "epochs" times
        for epoch in range(epochs):
            # randomly shuffle data
            if shuffle:
                x = np.random.permutation(x)
            for sample in x:
                # init the cluster weights for the first input vector
                if self.w is None and self.wg is None:
                    self._init_weights(x[0])
                    continue

                # global vector update
                self._update_global_weight(sample)

                # node activation & template matching
                activations = self._activation(sample)
                v_node_selection = np.argsort(activations)[::-1][:self.v]
                classes.append(v_node_selection[0])

                if train:
                    # check if resonance occurred
                    resonance = self._template_matching(sample, v_node_selection)
                    if resonance:
                        # update weight for the cluster
                        category = v_node_selection[0]
                        self.w[category] = self._update_weight(sample, self.w[category], self.lr)
                    else:
                        # no matching occurred
                        self._add_category(sample)

                    # group the v-nodes
                    if self.n_category > 1:
                        self._add_group(v_node_selection, sample, resonance)
                """
                print("After training the following parameters have been modified...")
                print("1. Weight Vector: ", self.w)
                print("2. Global Vector: ", self.wg)
                print("3. Number of Categories: ", self.n_category)
                print("4. Groups: ", self.group)
                """
        return self, classes

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
        x = np.atleast_2d(x)
        assert len(x.shape) == 2, "Wrong input vector shape: input.shape = (samples, features)"

        _, clustering_result = self.train(x, shuffle=False, train=train)
        return clustering_result


class DRNMAP(object):
    def __init__(self, lr=0.9, glr=1.0, alpha=1.0, rho=0.9, v=2):
        self.drn = DRN(lr, glr, alpha, rho, v)


if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


    def make_cluster_data():
        # TO DO: fixed mean, cov, num_cluster => parameterize!
        x, y = np.array([]), np.array([])
        mean = [[0.3, 0.2], [0.2, 0.7], [0.5, 0.5], [0.8, 0.4], [0.7, 0.8]]
        cov = [[0.001, 0], [0, 0.001]]
        for i in range(len(mean)):
            x_temp, y_temp = np.random.multivariate_normal(mean[i], cov, 30).T
            x = np.append(x, x_temp)
            y = np.append(y, y_temp)
        return x, y

    random.seed(43)
    print("TEST of DRN")
    data = io.loadmat('data.mat')['points']

    data_test = np.array([[4, 4],
                          [-2, -2],
                          [-5, 5],
                          [-2, 1.3],
                          [-1.5, -0.6],
                          [3, -1],
                          [-2.4, -1.5],
                          [-2.7, 3],
                          [2.9, -1.2],
                          [10, -10],
                          [-10, 10],
                          [5, 8]])
    
    drn = DRN(lr=1.0, rho=0.9)

    # data = data[:10]
    drn.train(data, shuffle=False)

    classes = np.array(drn.test(data))

    plt.figure(1)
    plt.plot(data[:, 0], data[:, 1], 'o')
    plt.title('Original data')

    plt.figure(2)
    for i in range(drn.n_category):
        plt.plot(data[classes == i, 0], data[classes == i, 1], 'x')
    # rect = patches.Rectangle((50, 100), 40, 30, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(
        plt.Rectangle((drn.wg[0][0], drn.wg[0][1]),
                      drn.wg[0][2] - drn.wg[0][0],
                      drn.wg[0][3] - drn.wg[0][1], fill=False,
                      edgecolor='r', linewidth=3)
    )
    plt.title('Classification result')

    plt.show()

    # sample1 = data[45]
    # dummy_w = np.hstack((data[:10], data[30:40]))
    # ff, bb = dummy_w[:, :drn.dim], dummy_w[:, drn.dim:]
    # ww1, ww2 = np.minimum(sample1, ff), np.maximum(sample1, bb)
    # drn.train(data)
    # print(drn.n_category)

    '''
    testART = DRN(rho=0.77)

    # training the FusionART
    x, y = make_cluster_data()
    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)

    classification_result_during_training = []
    classification_result_after_training = []
    for i in range(len(x)):
        _, tmp_class = testART.train(np.array([x[i], y[i]]))
        classification_result_during_training.append(tmp_class)

    classified_x, classified_y = [[np.array([]) for _ in range(testART.n_category)] for _ in range(2)]
    for i in range(len(x)):
        _, tmp_class = testART.train(np.array([x[i], y[i]]), train=False)
        tmp_class = tmp_class[0]
        classified_x[tmp_class] = np.append(classified_x[tmp_class], np.array([x[i]]))
        classified_y[tmp_class] = np.append(classified_y[tmp_class], np.array([y[i]]))
        classification_result_after_training.append(tmp_class)

    # print out the classification results
    plt.figure(1)
    plt.plot(x, y, 'x')
    plt.title('Original data')
    plt.axis([0, 1, 0, 1])

    plt.figure(2)
    for i in range(testART.n_category):
        plt.plot(classified_x[i], classified_y[i], 'x')
    plt.title('Classification result')
    plt.axis([0, 1, 0, 1])

    plt.show()
    '''
