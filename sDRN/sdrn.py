import numpy as np
from drn import DRN
import random
import warnings
from ..SFART.utils import l2_norm


class sDRN(DRN):
    def __init__(self, num_channel, input_dim, tmp_mat_elem, lr=0.9, glr=1.0, alpha=1.0, rho=0.7, v=2, gp=1, iov=0.85, dist=0.2):
        DRN.__init__(self, num_channel=num_channel, input_dim=input_dim, tmp_mat_elem=tmp_mat_elem, lr=lr, glr=glr, alpha=alpha, rho=rho, v=v)
        self.gp = gp  # probability for grouping
        self.iov = iov  # intersection of volume (IoV) condition for grouping two clusters
        self.dist = dist  # distance parameter

    def _grouping(self, idx):
        # find which cluster to group with idx-th cluster
        to_cluster, max_iov = None, 0
        for cluster in range(self.n_category):
            if cluster == idx:
                continue
            IoV, UoV = self._intersection_of_volume(self.w[cluster], self.w[idx])

            if all(UoV < self.dim * (1 - self.rho) * self._volume_of_cluster(self.wg)):
                distance = self._distance_between_clusters(self.w[cluster], self.w[idx])
                dist_glob = np.array([l2_norm(np.extract(self.wg[ch][self.dim:], self.wg[ch][:self.dim])) for ch in range(self.num_channel)])
                sum = np.sum(IoV)
                a = all(IoV > self.iov)
                b = sum > max_iov
                c = all(distance < np.multiply(self.dist, dist_glob))
                d = all(IoV > self.iov / 2)
                temp_cluster = self._union_of_clusters(self.w[idx], self.w[cluster])
                cluster_size_check = self._check_cluster_size_vig(temp_cluster)
                e = all(cluster_size_check)

                if (((a and b) or c) and d) and e:
                    to_cluster, max_iov = cluster, sum

        if to_cluster:
            self.n_category -= 1
            self.w[idx] = self._union_of_clusters(self.w[idx], self.w[to_cluster])
            self.w = np.delete(self.w, to_cluster, axis=0)

    def _volume_of_cluster(self, weight):
        front, back = self._split_weight_nch(weight)
        return np.prod(np.subtract(back, front), axis=1)

    def _union_of_clusters(self, w_i, w_j):
        front_i, back_i = self._split_weight_nch(w_i)
        front_j, back_j = self._split_weight_nch(w_j)
        u_front, u_back = np.minimum(front_i, front_j), np.maximum(back_i, back_j)
        return np.hstack((u_front, u_back))

    def _intersection_of_volume(self, w_i, w_j):
        w_i = w_i
        w_j = w_j
        volume1, volume2 = self._volume_of_cluster(w_i), self._volume_of_cluster(w_j)
        union_weight = self._union_of_clusters(w_i, w_j)
        union_volume = self._volume_of_cluster(union_weight)
        return np.divide(np.add(volume1, volume2), union_volume), np.array(union_volume)

    def _distance_between_clusters(self, w_i, w_j):
        front_i, back_i = self._split_weight_nch(w_i)
        front_j, back_j = self._split_weight_nch(w_j)
        size_i, size_j = l2_norm(np.subtract(back_i, front_i)) / 2, l2_norm(np.subtract(back_j, front_j)) / 2
        dist_from_center = l2_norm(np.subtract(np.add(front_i, back_i) / 2, np.add(front_j, back_j) / 2))
        distance = np.maximum(np.subtract(np.subtract(dist_from_center, size_i), size_j), np.zeros(self.num_channel))
        return np.array(distance)

    def _distance_between_cluster_and_point(self, weight, sample):
        front, back = self._split_weight_nch(weight)
        size = l2_norm(np.subtract(back, front)) / 2
        distance_from_center = l2_norm(np.subtract(sample, np.add(front, back) / 2))
        distance = np.maximum(np.subtract(distance_from_center, size), np.zeros(self.num_channel))
        return np.array(distance)

    def _learning_condition(self, sample, idx):
        weight = self.w[idx]
        volume_orig = self._volume_of_cluster(weight)
        adaptive_lr = 2 * np.divide(volume_orig, (self.dim * (1 - self.rho) * self._volume_of_cluster(self.wg)))
        adaptive_lr = np.minimum(adaptive_lr, self.lr)
        if adaptive_lr ==0:
            adaptive_lr=0.1
        dist_glob = np.array([l2_norm(np.subtract(self.wg[ch][self.dim:], self.wg[ch][:self.dim])) for ch in range(self.num_channel)])
        condition = self._distance_between_cluster_and_point(weight, sample) < self.dist * dist_glob
        return np.array(condition), np.array(adaptive_lr)

    def train(self, x, epochs=1, shuffle=True, train=True):
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
                activations = self._rdrn_activation(sample)
                v_node_selection = np.argsort(activations)[::-1][:self.v]
                classes.append(v_node_selection[0])

                if train:
                    # check if resonance occurred
                    match_val = self._template_matching(sample, v_node_selection[0])
                    condition, adaptive_lr = self._learning_condition(sample, v_node_selection[0])
                    if all(match_val) and all(condition):
                        # update weight for the cluster
                        category = v_node_selection[0]
                        self.w[category] = self._update_weight(sample, self.w[category], 0.1)
                    else:
                        # no matching occurred
                        self._add_category(sample)
                    # randomly incur cluster grouping
                    if random.uniform(0, 1) < self.gp:
                        if all(match_val) and all(condition):
                            self._grouping(category)
                        else:
                            self._grouping(self.n_category - 1)
        return classes
