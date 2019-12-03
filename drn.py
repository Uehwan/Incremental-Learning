import numpy as np
from functools import partial
import random
import matplotlib.pyplot as plt
import scipy.io as io

import warnings
l2_norm = partial(np.linalg.norm, ord=2, axis=-1)
warnings.simplefilter(action='ignore', category=FutureWarning)

class DRN(object):
    def __init__(self, num_channel, input_dim, tmp_mat_elem, lr=0.9, glr=1.0, alpha=1.0, rho=0.7, v=2):
        self.lr = lr  # learning rate
        self.glr = glr  # global learning rate
        self.alpha = alpha  # parameter for node activation
        self.rho = rho  # rho parameter
        self.v = v  # number of nodes to select

        self.w = None  # weights for clusters; (samples, weights)
        self.wg = None  # global weight vector
        self.dim = None  # dimension of input vectors
        self.n_category = 0  # number of categories
        self.group = {}  # group container

        self.num_channel = num_channel
        self.input_dim = input_dim
        self.tmp_mat_elem = tmp_mat_elem
        assert num_channel == len(input_dim), "num_channel should be array length of input_dim"

    #    Utils functions
    def extract_append(self, array, vector, ch):
        temp = []
        for i in range(len(vector)):
            temp.append(array[vector[i]][ch])
        return np.array(temp)

    def _distance(self, sample_ch, weight_ch):
        f, b, w1, w2 = self._split_weight_ch(weight_ch, sample_ch)
        distance = l2_norm(w1 - f + w2 - b)
        return distance

    def _split_weight_ch(self, weight_ch, sample_ch=None):
        # weight_ch and sample_ch should be 1D.
        if sample_ch is None:
            front, back = weight_ch[:self.dim], weight_ch[self.dim:]
            return np.array(front), np.array(back)
        else:
            front, back = weight_ch[:self.dim], weight_ch[self.dim:]
            w1, w2 = np.minimum(sample_ch, front), np.maximum(sample_ch, back)
            return np.array(front), np.array(back), np.array(w1), np.array(w2)

    def _split_weight_nch(self, weight, sample=None):
        # Weight is 2D, [ch][feature]
        # Sample is 1D, [feature] (within [ch])

        if sample is None:
            front_list = [weight[ch][:self.dim] for ch in range(len(weight))]
            back_list = [weight[ch][self.dim:] for ch in range(len(weight))]
            return np.array(front_list), np.array(back_list)
        else:
            front_list = [weight[ch][:self.dim] for ch in range(len(weight))]
            back_list = [weight[ch][self.dim:] for ch in range(len(weight))]
            w1_list = np.minimum(sample, front_list)
            w2_list = np.minimum(sample, back_list)
            return np.array(front_list), np.array(back_list), np.array(w1_list), np.array(w2_list)

    # Weight related functions
    def _init_weights(self, sample):

        self.w = np.atleast_2d([np.hstack((sample, sample))])
        self.wg = np.atleast_2d(np.hstack((sample, sample)))
        self.dim = self.input_dim[0]
        self.n_category += 1

    # New node resonance related functions
    def _drn_activation(self, sample):
        activation = []
        for category in range(self.n_category):
            temp_activation = np.sum([np.exp(-self.alpha * self._distance(sample[ch], self.w[category][ch])) for ch in
                                      range(self.num_channel)])
            activation.append(temp_activation)
        return np.array(activation)

    def _rdrn_activation(self, sample):
        activation = []
        for category in range(self.n_category):
            dist_glob = np.array([l2_norm(np.subtract(self.wg[ch][self.dim:], self.wg[ch][:self.dim])) for ch in
                                  range(self.num_channel)])
            temp_activation = np.sum(
                [np.exp(-self.alpha * self._distance(sample[ch], self.w[category][ch]) / dist_glob[ch]) for ch in
                 range(self.num_channel)])
            activation.append(temp_activation)
        return np.array(activation)

    def _template_matching(self, sample, category):
        match_val = []
        for ch in range(self.num_channel):
            front, back, _, _ = self._split_weight_ch(self.wg[ch], sample[ch])
            _, _, w1, w2 = self._split_weight_ch(self.w[category][ch], sample[ch])

            if self.tmp_mat_elem is True:
                M = np.abs(back - front)
                S = np.abs(w2 - w1)
                match_val.append(np.sum((M - S) / M) > self.rho)
            else:
                M = np.sum(np.abs(back - front))
                S = np.sum(np.abs(w2-w1))
                match_val.append((M - S) / M > self.rho)
        return np.array(match_val)

    def _check_cluster_size_vig(self, cluster):
        temp_node = np.array([cluster[ch][:self.dim] for ch in range(self.num_channel)])
        match_val = []
        for ch in range(self.num_channel):
            front, back, _, _ = self._split_weight_ch(self.wg[ch], temp_node[ch])
            _, _, w1, w2 = self._split_weight_ch(cluster[ch], temp_node[ch])

            if self.tmp_mat_elem is True:
                M = np.abs(back - front)
                S = np.abs(w2 - w1)
                match_val.append(np.sum((M - S) / M) > self.rho)
            else:
                M = np.sum(np.abs(back - front))
                S = np.sum(np.abs(w2-w1))
                match_val.append((M - S) / M > self.rho)
        return np.array(match_val)

    def _add_category(self, sample):
        self.n_category += 1
        new_weight = np.hstack((sample, sample))
        self.w = np.vstack((self.w, np.array([new_weight])))

    def _update_global_weight(self, sample, grouping=True):
        for ch in range(self.num_channel):
            self.wg = self._update_weight(sample, self.wg, self.glr)
        if len(self.group) > 0 and grouping:
            self._grouping()

    def _update_weight(self, sample, weight, lr):
        # Mind the dimension
        # It is [input_dim], both sample and weight.
        w1_list = []
        w2_list = []
        for ch in range(self.num_channel):
            a = self._split_weight_ch(weight[ch], sample[ch])
            b = self._split_weight_ch(weight[ch], sample[ch])
            w1_list.append(a[2])
            w2_list.append(b[3])
        #w1_list = [self._split_weight_ch(weight[ch], sample[ch])[2] for ch in range(self.num_channel)]
        #w2_list = [self._split_weight_ch(weight[ch], sample[ch])[3] for ch in range(self.num_channel)]
        if np.isscalar(lr):
            updated_weight = lr * np.hstack((w1_list, w2_list)) + (1 - lr) * np.array(weight)
        else:
            updated_weight = np.add(np.multiply(lr, np.hstack((w1_list, w2_list))),
                                    np.multiply((1 - lr), np.array(weight)))
        return np.array(updated_weight)

    # Grouping related functions
    def _grouping(self):
        while True:
            resonance, idx_s, idx_l, w_ij_1_list, w_ij_2_list = self._condition_for_grouping()
            if not resonance:
                break

            # merge two clusters
            self.w[idx_s] = np.array([np.hstack((w_ij_1_list[ch], w_ij_2_list[ch])) for ch in range(self.num_channel)])

            # reconnect nodes previously connected to "idx_l" to "idx_s"
            to_delete_group, to_add_group = [], []
            for check in self.group:
                if idx_l in check:
                    item1, item2 = check
                    reconnection = {item1, item2}
                    _, _ = reconnection.remove(idx_l), reconnection.add(idx_s)
                    reconnection = sorted(reconnection)
                    to_delete_group.append((item1, item2))  # self-connection considered
                    if len(reconnection) == 2:
                        # update the synaptic strength
                        item1, item2 = reconnection
                        T = 0
                        for ch in range(self.num_channel):
                            subtraction = np.array(self.w[item1][ch] - self.w[item2][ch])
                            center_of_mass_diff = (subtraction[:self.dim] + subtraction[self.dim:]) / 2
                            T += np.exp(-self.alpha * l2_norm(center_of_mass_diff))
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
        shuffled_keys = random.sample(list(self.group.keys()), len(list(self.group.keys())))
        for idx_s, idx_l in shuffled_keys:  # self.group:
            resonance, w_ij_1_list, w_ij_2_list = self._resonance_between_clusters(idx_s, idx_l)
            if resonance:
                return resonance, idx_s, idx_l, w_ij_1_list, w_ij_2_list
        return False, 0, 0, 0, 0

    def _resonance_between_clusters(self, idx_s, idx_l):

        front = np.array([self.wg[ch][:self.dim] for ch in range(self.num_channel)])
        back = np.array([self.wg[ch][self.dim:] for ch in range(self.num_channel)])
        M = np.sum(np.abs(np.subtract(back, front)), axis=1)
        w_i_list = [np.array(self.w[idx_s][ch]) for ch in range(self.num_channel)]
        w_j_list = [np.array(self.w[idx_l][ch]) for ch in range(self.num_channel)]
        w_i_front_list, w_i_back_list = self._split_weight_nch(w_i_list)
        w_j_front_list, w_j_back_list = self._split_weight_nch(w_j_list)
        w_ij_front_list, w_ij_back_list = np.minimum(w_i_front_list, w_j_front_list), np.maximum(w_i_back_list,
                                                                                                 w_j_back_list)
        S = np.sum(np.abs(np.subtract(w_ij_back_list, w_ij_front_list)), axis=1)
        resonance_list = np.divide(np.subtract(M, S), M) > self.rho

        return all(resonance_list), np.array(w_ij_front_list), np.array(w_ij_back_list)

    def _add_group(self, v_nodes, sample, condition):
        # See the paper. front/back/center_of_mass should be vector
        # Mixed and calculate from vector to T in the T = np.exp ~~~ thing. (Make ch loop there or vectorization.)
        center_of_mass_list = []
        if all(condition):
            to_connect = np.copy(v_nodes)
            front = np.array([self._split_weight_nch(self.extract_append(self.w, v_nodes, ch), sample[ch])[0] for ch in
                              range(self.num_channel)])
            back = np.array([self._split_weight_nch(self.extract_append(self.w, v_nodes, ch), sample[ch])[1] for ch in
                             range(self.num_channel)])
            center_of_mass_list = (front + back) / 2

        else:

            to_connect = np.copy(v_nodes)
            to_connect = np.hstack((to_connect, self.n_category - 1))

            front = np.array([self._split_weight_nch(self.extract_append(self.w, v_nodes, ch), sample[ch])[0] for ch in
                              range(self.num_channel)])
            back = np.array([self._split_weight_nch(self.extract_append(self.w, v_nodes, ch), sample[ch])[1] for ch in
                             range(self.num_channel)])
            center_of_mass_list = (front + back) / 2
            sample_list = np.array([sample for ch in range(self.num_channel)])
            center_of_mass_list = np.concatenate((center_of_mass_list, sample_list), axis=1)

        for first in range(len(to_connect)):
            for second in range(first + 1, len(to_connect)):
                smaller, larger = sorted([to_connect[first], to_connect[second]])
                # new connections get added (first condition)
                # and synaptic strengths get updated (second condition)
                T = np.sum(
                    [np.exp(-self.alpha * l2_norm(center_of_mass_list[ch][first] - center_of_mass_list[ch][second])) for
                     ch in range(self.num_channel)])

                if not T == 0 or v_nodes[0] in (smaller, larger):
                    self.group[(smaller, larger)] = T

    # Train, test functions
    def train(self, x, epochs=1, shuffle=False, train=True):

        classes = []

        if shuffle:
            x = np.random.permutation(x)

        i = 0
        for sample in x:
            # init the cluster weights for the first input vector
            if self.w is None and self.wg is None:
                self._init_weights(sample)
                continue
            i += 1
#            if i % 200 == 0:
#                print(i, "th iteration ")
            # global vector update
            self._update_global_weight(sample)

            # node activation & template matching
            activations = self._drn_activation(sample)
            v_node_selection = np.argsort(activations)[::-1][:self.v]
            classes.append(v_node_selection[0])

            if train:
                ##Here we should add ch for loop, (opinion not confirmed)
                # check if resonance occurred
                match_val = self._template_matching(sample, v_node_selection[0])
                if all(match_val):
                    # update weight for the cluster
                    category = v_node_selection[0]
                    self.w[category] = self._update_weight(sample, self.w[category], self.lr)
                else:
                    # no matching occurred
                    self._add_category(sample)

                # connect the v-nodes
                if self.n_category > 1:
                    self._add_group(v_node_selection, sample, match_val)
        return classes

    def test(self, x, train=False):
        clustering_result = self.train(x, shuffle=False, train=train)
        return clustering_result
