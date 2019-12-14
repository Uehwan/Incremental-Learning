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
          Neural Networks, Dec. 2018, Submitted.
        > Implementation of DRNMAP algorithm and performance analysis
        > DRNMAP employs rDRN rather than the original DRN
        > rDRN stands for reduced-DRN, which uses improved grouping
"""

import numpy as np
from functools import partial

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

    def _update_global_weight(self, sample, grouping=True):
        if self._distance(sample, self.wg):
            self.wg = self._update_weight(sample, self.wg, self.glr)
            if len(self.group) > 0 and grouping:
                self._grouping()

    def _update_weight(self, sample, weight, lr):
        _, _, w1, w2 = self._split_weight(weight, sample)
        updated_weight = lr * np.hstack((w1, w2)) + (1 - lr) * weight
        return updated_weight

    def _split_weight(self, weight, sample=None):
        weight = np.atleast_2d(weight)
        front, back = weight[:, :self.dim], weight[:, self.dim:]
        if sample is not None:
            w1, w2 = np.minimum(sample, front), np.maximum(sample, back)
            return front, back, w1, w2
        return front, back

    def _grouping(self):
        while True:
            resonance, idx_s, idx_l, w_ij_1, w_ij_2 = self._condition_for_grouping()
            if not resonance:
                break

            # merge two clusters
            self.w[idx_s] = np.hstack((w_ij_1, w_ij_2))
            to_delete_group, to_add_group = [], []

            # reconnect nodes previously connected to "idx_l" to "idx_s"
            for check in self.group:
                if idx_l in check:
                    item1, item2 = check
                    reconnection = {item1, item2}
                    _, _ = reconnection.remove(idx_l), reconnection.add(idx_s)
                    reconnection = sorted(reconnection)
                    to_delete_group.append((item1, item2)) # self-connection considered
                    if len(reconnection) == 2:
                        # update the synaptic strength
                        item1, item2 = reconnection
                        subtraction = np.atleast_2d(self.w[item1] - self.w[item2])
                        center_of_mass_diff = (subtraction[:, :self.dim] + subtraction[:, self.dim:]) / 2
                        T = np.exp(-self.alpha * l2_norm(center_of_mass_diff))
                        to_add_group.append([(item1, item2), T])

            # delete the collected items from group and update w and n_category
            self._update_groups(to_add_group, to_delete_group)
            self.w = np.delete(self.w, idx_l, axis=0)
            self.n_category -= 1

            # update indices > idx_l
            _, _ = to_delete_group.clear(), to_add_group.clear()

            for check in self.group:
                if any([c > idx_l for c in check]):
                    item1, item2 = check
                    residuals = [-1 if item > idx_l else 0 for item in check]
                    item1, item2 = item1 + residuals[0], item2 + residuals[1]
                    to_add_group.append([(item1, item2), self.group[check]])
                    to_delete_group.append(check)

            self._update_groups(to_add_group, to_delete_group)

    def _update_groups(self, to_add, to_delete):
        for pair, strength in to_add:
            self.group[pair] = strength
        for delete in to_delete:
            del self.group[delete]

    def _condition_for_grouping(self):
        shuffled_keys = random.shuffle(list(self.group.keys()))
        for idx_s, idx_l in shuffled_keys:  # self.group:
            resonance, w_ij_1, w_ij_2 = self._resonance_between_clusters(idx_s, idx_l)
            if resonance:
                return resonance, idx_s, idx_l, w_ij_1, w_ij_2
        return False, 0, 0, 0, 0

    def _resonance_between_clusters(self, idx_s, idx_l):
        front, back = self.wg[:, :self.dim], self.wg[:, self.dim:]
        M = np.sum(np.abs(back - front))
        w_i, w_j = np.atleast_2d(self.w[idx_s]), np.atleast_2d(self.w[idx_l])
        w_i_front, w_i_back = self._split_weight(w_i)
        w_j_front, w_j_back = self._split_weight(w_j)
        w_ij_front, w_ij_back = np.minimum(w_i_front, w_j_front), np.maximum(w_i_back, w_j_back)
        S = np.sum(np.abs(w_ij_back - w_ij_front))
        return (M - S) / M > self.rho, w_ij_front, w_ij_back

    def _distance(self, sample, weight):
        f, b, w1, w2 = self._split_weight(weight, sample)
        distance = l2_norm(w1 - f + w2 - b)
        return distance

    def _activation(self, sample):
        distance = self._distance(sample, self.w)
        activation = np.exp(-self.alpha * distance)
        return activation

    def _template_matching(self, sample, v_nodes):
        front, back, _, _ = self._split_weight(self.wg, sample)
        M = np.sum(np.abs(back - front))
        _, _, w1, w2 = self._split_weight(self.w[v_nodes[0]], sample)
        S = np.sum(np.abs(w2 - w1))
        return (M - S) / M > self.rho

    def _add_category(self, sample):
        self.n_category += 1
        new_weight = np.hstack((sample, sample))
        self.w = np.vstack((self.w, new_weight))

    def _add_group(self, v_nodes, sample, condition):
        front, back, _, _ = self._split_weight(self.w[v_nodes], sample)
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
            - classes: the categories of each sample
        """
        # check if the input vectors (x) are of shape (samples, features)
        x = np.atleast_2d(x)
        assert len(x.shape) == 2, "Wrong input vector shape: input.shape = (samples, features)"

        classes = []
        # training for "epochs" times
        for epoch in range(epochs):
            if shuffle:
                # randomly shuffle data
                x = np.random.permutation(x)
            for sample in x:
                # init the cluster weights for the first input vector
                if self.w is None and self.wg is None:
                    self._init_weights(sample)
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

                    # connect the v-nodes
                    if self.n_category > 1:
                        self._add_group(v_node_selection, sample, resonance)
        return classes

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

        clustering_result = self.train(x, shuffle=False, train=train)
        return clustering_result


class rDRN(DRN):
    def __init__(self, lr=0.9, glr=1.0, alpha=1.0, rho=0.9, v=1, gp=0.75, iov=0.85, dist=0.2):
        DRN.__init__(self, lr=lr, glr=glr, alpha=alpha, rho=rho, v=v)
        self.gp = gp      # probability for grouping
        self.iov = iov    # intersection of volume (IoV) condition for grouping two clusters
        self.dist = dist  # distance parameter

    def _grouping(self, idx):
        # find which cluster to group with idx-th cluster
        to_cluster, max_iov = None, 0
        for cluster in range(self.n_category):
            if cluster == idx:
                continue
            IoV, UoV = self._intersection_of_volume(self.w[cluster], self.w[idx])
            if UoV < self.dim * (1 - self.rho) * self._volume_of_cluster(self.wg):
                distance = self._distance_between_clusters(self.w[cluster], self.w[idx])
                dist_glob = l2_norm(self.wg[:, self.dim:] - self.wg[:, :self.dim])
                if IoV > self.iov and IoV > max_iov or distance < self.dist * dist_glob and IoV > self.iov / 2:
                    to_cluster, max_iov = cluster, IoV

        if to_cluster:
            self.n_category -= 1
            self.w[cluster] = self._union_of_clusters(self.w[idx], self.w[to_cluster])
            self.w = np.delete(self.w, to_cluster, axis=0)

    def _volume_of_cluster(self, weight):
        weight = np.atleast_2d(weight)
        front, back = self._split_weight(weight)
        return np.prod(back - front)

    def _union_of_clusters(self, w_i, w_j):
        w_i, w_j = np.atleast_2d(w_i), np.atleast_2d(w_j)
        front_i, back_i = self._split_weight(w_i)
        front_j, back_j = self._split_weight(w_j)
        u_front, u_back = np.minimum(front_i, front_j), np.maximum(back_i, back_j)
        return np.hstack((u_front, u_back))

    def _intersection_of_volume(self, w_i, w_j):
        volume1, volume2 = self._volume_of_cluster(w_i), self._volume_of_cluster(w_j)
        union_weight = self._union_of_clusters(w_i, w_j)
        union_volume = self._volume_of_cluster(union_weight)
        return (volume1 + volume2) / union_volume, union_volume

    def _distance_between_clusters(self, w_i, w_j):
        w_i, w_j = np.atleast_2d(w_i), np.atleast_2d(w_j)
        front_i, back_i = self._split_weight(w_i)
        front_j, back_j = self._split_weight(w_j)
        size_i, size_j = l2_norm(back_i - front_i) / 2, l2_norm(back_j - front_j) / 2
        dist_from_center = l2_norm((front_i + back_i) / 2 - (front_j + back_j) / 2)
        distance = max(dist_from_center - size_i - size_j, 0)
        return distance

    def _distance_between_cluster_and_point(self, weight, sample):
        weight = np.atleast_2d(weight)
        front, back = self._split_weight(weight)
        size = l2_norm(back - front) / 2
        distance_from_center = l2_norm(sample - (front + back) / 2)
        distance = max(distance_from_center - size, 0)
        return distance

    def _learning_condition(self, sample, idx):
        weight = self.w[idx]
        volume_orig = self._volume_of_cluster(weight)
        adaptive_lr = 2 * volume_orig / (self.dim * (1 - self.rho) * self._volume_of_cluster(self.wg))
        adaptive_lr = min(adaptive_lr, self.lr)
        dist_glob = l2_norm(self.wg[:, self.dim:] - self.wg[:, :self.dim])
        condition = self._distance_between_cluster_and_point(weight, sample) < self.dist * dist_glob
        return condition, adaptive_lr

    def train(self, x, epochs=1, shuffle=True, train=True):
        """
        Description
            - Train rDRN model with the input vectors
            - rDRN automatically clusters the input vectors

        Parameters
            - x: 2d array of size (samples, features), where all features can
                 range [-inf, inf]
            - epochs: the number of training loop
            - permute: whether to shuffle the x's when training

        Return
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
                    self._init_weights(sample)
                    continue

                # global vector update without grouping
                self._update_global_weight(sample, True)

                # node activation & template matching
                activations = self._activation(sample)
                v_node_selection = np.argsort(activations)[::-1][:self.v]
                classes.append(v_node_selection[0])

                if train:
                    # check if resonance occurred
                    resonance = self._template_matching(sample, v_node_selection)
                    condition, adaptive_lr = self._learning_condition(sample, v_node_selection[0])
                    if resonance and condition:
                        # update weight for the cluster
                        category = v_node_selection[0]
                        self.w[category] = self._update_weight(sample, self.w[category], adaptive_lr)
                    else:
                        # no matching occurred
                        self._add_category(sample)

                    # randomly incur cluster grouping
                    if random.uniform(0, 1) < self.gp:
                        if resonance and condition:
                            self._grouping(category)
                        else:
                            self._grouping(self.n_category - 1)
        return classes


class DRNMAP(rDRN):
    def __init__(self, lr=0.9, glr=1.0, epsilon=0.001, alpha=1.0, rho=0.9, v=1, gp=0.75, iov=0.85, dist=0.2):
        rDRN.__init__(self, lr=lr, glr=glr, alpha=alpha, rho=rho, v=v, gp=gp, iov=iov, dist=dist)
        self.map = []
        self.epsilon = epsilon

    def _init_weights(self, sample, label):
        super(DRNMAP, self)._init_weights(sample)
        self.map.append(label)

    def _add_category(self, sample, label):
        super(DRNMAP, self)._add_category(sample)
        self.map.append(label)

    def _grouping(self, idx):
        # find which cluster to group with idx-th cluster
        to_cluster, max_iov = None, 0
        for cluster in range(self.n_category):
            if cluster == idx:
                continue
            IoV, UoV = self._intersection_of_volume(self.w[cluster], self.w[idx])
            if UoV < self.dim * (1 - self.rho) * self._volume_of_cluster(self.wg):
                distance = self._distance_between_clusters(self.w[cluster], self.w[idx])
                dist_glob = l2_norm(self.wg[:, self.dim:] - self.wg[:, :self.dim])
                if IoV > self.iov and IoV > max_iov or distance < self.dist * dist_glob and IoV > self.iov / 2:
                    to_cluster, max_iov = cluster, IoV

        condition = self._grouping_condition(idx, to_cluster)

        if to_cluster and condition:
            self.n_category -= 1
            self.w[cluster] = self._union_of_clusters(self.w[idx], self.w[to_cluster])
            self.w = np.delete(self.w, to_cluster, axis=0)
            del self.map[to_cluster]

    def _grouping_condition(self, idx, to_cluster):
        if self.map[idx] == self.map[to_cluster]:
            return True
        else:
            # need to check neighboring clusters
            return False

    def _template_matching_for_all(self, sample):
        front, back, _, _ = self._split_weight(self.wg, sample)
        M = np.sum(np.abs(back - front))
        resonance_values = []
        for i in range(self.n_category):
            _, _, w1, w2 = self._split_weight(self.w[v_nodes[0]], sample)
            S = np.sum(np.abs(w2 - w1))
            resonance_values.append((M - S) / M)
        return np.array(resonance_values)

    def _match_category(self, sample, label=None):
        _rho = self.rho
        # check activation values
        scores = self._activation(sample)
        # check resonance values
        norms = self._template_matching_for_all(sample)

        threshold = norms >= _rho
        while not np.all(threshold == False):
            y_ = np.argmax(scores * threshold.astype(int))

            if label is None or self.map[y_] == label:
                return map[y_]
            else:
                _rho = norms[y_] + self.epsilon
                norms[y_] = 0
                threshold = norms >= _rho
        return -1

    def train(self, x, y, epochs=1, shuffle=True):
        """
        Description
            - Train rDRN model with the input vectors
            - rDRN automatically clusters the input vectors

        Parameters
            - x: 2d array of size (samples, features), where all features can
                 range [-inf, inf]
            - y: labels for x
            - epochs: the number of training loop
            - permute: whether to shuffle the x's when training

        Return
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
                x, y = zip(*np.random.permutation(zip(x, y)))

            for sample, label in zip(x, y):
                # init the cluster weights for the first input vector
                if self.w is None and self.wg is None:
                    self._init_weights(sample, label)
                    continue

                # global vector update without grouping
                self._update_global_weight(sample, False)

                # match tracking
                classification = self._match_tracking(sample, label)
                classes.append(classification)

                # check if matched successfully
                if classification == -1:
                    # no matching occurred
                    self._add_category(sample, label)
                else:
                    # check if resonance occurred
                    resonance = self._template_matching(sample, [classification])
                    condition, adaptive_lr = self._learning_condition(sample, classification)
                    if resonance and condition:
                        # update weight for the cluster
                        self.w[classification] = self._update_weight(sample, self.w[classification], adaptive_lr)
                    else:
                        self._add_category(sample, label)

                # randomly incur cluster grouping
                if random.uniform(0, 1) < self.gp:
                    if resonance and condition:
                        self._grouping(classification)
                    else:
                        self._grouping(self.n_category - 1)
        return classes

    def test(self, x):
        """
        Description
            - Test the trained DRN model with the input vectors
            - DRN predicts the clusters of the input vectors

        Parameters
            - x: 2d array of size (samples, features), where all features can
                 range [-inf, inf]

        Return
            - labels: DRN class itself
        """
        x = np.atleast_2d(x)
        assert len(x.shape) == 2, "Wrong input vector shape: input.shape = (samples, features)"

        labels = np.zeros(len(x))
        for i, sample in enumerate(x):
            category = self._match_category(sample)
            labels[i] = self.map[category]
        return labels


if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt
    import scipy.io as io


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
    
    drn = rDRN(lr=0.8, rho=0.75)

    # data = data[:10]
    drn.train(data, shuffle=True)

    classes = np.array(drn.test(data))

    # plt.figure(1)
    # plt.plot(data[:, 0], data[:, 1], 'o')
    # plt.title('Original data')

    plt.figure()
    for i in range(drn.n_category):
        plt.plot(data[classes == i, 0], data[classes == i, 1], 'x')
        plt.gca().add_patch(
            plt.Rectangle((drn.w[i][0], drn.w[i][1]),
                          drn.w[i][2] - drn.w[i][0],
                          drn.w[i][3] - drn.w[i][1], fill=False,
                          edgecolor='b', linewidth=3)
        )
    # rect = patches.Rectangle((50, 100), 40, 30, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(
        plt.Rectangle((drn.wg[0][0], drn.wg[0][1]),
                      drn.wg[0][2] - drn.wg[0][0],
                      drn.wg[0][3] - drn.wg[0][1], fill=False,
                      edgecolor='r', linewidth=3)
    )
    plt.title('Classification result')
    # plt.axis([-15, 15, -15, 15])

    plt.show()

    # sample1 = data[45]
    # dummy_w = np.hstack((data[:10], data[30:40]))
    # ff, bb = dummy_w[:, :drn.dim], dummy_w[:, drn.dim:]
    # ww1, ww2 = np.minimum(sample1, ff), np.maximum(sample1, bb)
    # drn.train(data)
    # print(drn.n_category)

    testART = rDRN(lr=0.9, rho=0.8)

    # training the FusionART
    x, y = make_cluster_data()
    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)

    classification_result_during_training = []
    classification_result_after_training = []
    for i in range(len(x)):
        tmp_class = testART.train(np.array([x[i], y[i]]), shuffle=False)
        classification_result_during_training.append(tmp_class)

    classified_x, classified_y = [[np.array([]) for _ in range(testART.n_category)] for _ in range(2)]
    for i in range(len(x)):
        tmp_class = testART.train(np.array([x[i], y[i]]), train=False)
        tmp_class = tmp_class[0]
        classified_x[tmp_class] = np.append(classified_x[tmp_class], np.array([x[i]]))
        classified_y[tmp_class] = np.append(classified_y[tmp_class], np.array([y[i]]))
        classification_result_after_training.append(tmp_class)

    # print out the classification results
    # plt.figure()
    # plt.plot(x, y, 'x')
    # plt.title('Original data')
    # plt.axis([0, 1, 0, 1])

    plt.figure()
    for i in range(testART.n_category):
        plt.plot(classified_x[i], classified_y[i], 'x')
        plt.gca().add_patch(
            plt.Rectangle((testART.w[i][0], testART.w[i][1]),
                          testART.w[i][2] - testART.w[i][0],
                          testART.w[i][3] - testART.w[i][1], fill=False,
                          edgecolor='b', linewidth=3)
        )
    plt.gca().add_patch(
        plt.Rectangle((testART.wg[0][0], testART.wg[0][1]),
                      testART.wg[0][2] - testART.wg[0][0],
                      testART.wg[0][3] - testART.wg[0][1], fill=False,
                      edgecolor='r', linewidth=3)
    )
    plt.title('Classification result')
    plt.axis([0, 1, 0, 1])

    plt.show()

    #############################################
    ### DRNMAP Test Code
    #############################################
    from sklearn.svm import SVC
    from sklearn import datasets


    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # 한글출력
    plt.rcParams['axes.unicode_minus'] = False

    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # flower length & width
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    # SVM 분류 모델
    svm_clf = SVC(kernel="linear", C=float("inf"))
    svm_clf.fit(X, y)

    x0 = np.linspace(0, 5.5, 200)
    pred_1 = 5 * x0 - 20
    pred_2 = x0 - 1.8
    pred_3 = 0.1 * x0 + 0.5


    def plot_svc_decision_boundary(svm_clf, xmin, xmax):
        w = svm_clf.coef_[0]
        b = svm_clf.intercept_[0]

        # 결정 경계에서 w0*x0 + w1*x1 + b = 0 이므로
        # => x1 = -w0/w1 * x0 - b/w1
        x0 = np.linspace(xmin, xmax, 200)
        decision_boundary = -w[0] / w[1] * x0 - b / w[1]

        margin = 1 / w[1]
        gutter_up = decision_boundary + margin
        gutter_down = decision_boundary - margin

        svs = svm_clf.support_vectors_
        plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
        plt.plot(x0, decision_boundary, "k-", linewidth=2)
        plt.plot(x0, gutter_up, "k--", linewidth=2)
        plt.plot(x0, gutter_down, "k--", linewidth=2)

    plt.figure(figsize=(12, 2.7))

    plt.subplot(121)
    plt.plot(x0, pred_1, "g--", linewidth=2)
    plt.plot(x0, pred_2, "m-", linewidth=2)
    plt.plot(x0, pred_3, "r-", linewidth=2)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris-Setosa")
    plt.xlabel("Flower Length", fontsize=14)
    plt.ylabel("Flowe Width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.subplot(122)
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo")
    plt.xlabel("Flower Length", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.show()
    Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
    ys = np.array([0, 0, 1, 1])
    svm_clf = SVC(kernel="linear", C=100)
    svm_clf.fit(Xs, ys)

    plt.figure(figsize=(12, 3.2))
    plt.subplot(121)
    plt.plot(Xs[:, 0][ys == 1], Xs[:, 1][ys == 1], "bo")
    plt.plot(Xs[:, 0][ys == 0], Xs[:, 1][ys == 0], "ms")
    plot_svc_decision_boundary(svm_clf, 0, 6)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel("$x_1$  ", fontsize=20, rotation=0)
    plt.title("Before Scale Adjustment", fontsize=16)
    plt.axis([0, 6, 0, 90])

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xs)
    svm_clf.fit(X_scaled, ys)

    plt.subplot(122)
    plt.plot(X_scaled[:, 0][ys == 1], X_scaled[:, 1][ys == 1], "bo")
    plt.plot(X_scaled[:, 0][ys == 0], X_scaled[:, 1][ys == 0], "ms")
    plot_svc_decision_boundary(svm_clf, -2, 2)
    plt.xlabel("$x_0$", fontsize=20)
    plt.title("After Scale Adjustment", fontsize=16)
    plt.axis([-2, 2, -2, 2])

    plt.show()
