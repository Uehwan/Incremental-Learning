import numpy as np
from utils import fuzz_min_sum


class FusionART:
    """
    Fusion ART based on Fuzzy ART
    - G. A. Carpenter, S. Grossberg, and D. B. Rosen, “Fuzzy ART: An adaptive resonance
    algorithm for rapid, stable classification of analog patterns,”
    in Proc. Int. Joint Conf. Neural Netw., vol. 2, 1991, pp. 411–416.

    """

    def __init__(self, num_channel, input_dim, complement_coding=True, gamma=0.01,
                 alpha=0.5, rho=0.9, contribution_param=""):

        self.num_channel = num_channel
        self.input_dim = input_dim
        self.complement_coding = complement_coding
        self.gamma = gamma
        self.alpha = alpha
        self.rho_default = rho
        self.last_retrieved_memory = 0

        if contribution_param:
            assert len(contribution_param) == len(input_dim), "contribution parameters not well specified!"
            self.contribution_param = contribution_param
        else:
            # TO DO
            # contribution param: number => list
            # current implementation: channels are equally contributing
            self.contribution_param = 1 / self.num_channel

        self.n_category = 1
        # for number of categories => create 1) weight, 2) rho parameter, 3) memory strength
        if self.complement_coding:
            self.w = [[np.append(np.ones(ni), np.zeros(ni)) for ni in input_dim]]
        else:
            self.w = [[np.ones(ni) for ni in input_dim]]

        # rho parameter for each category
        self.rho = np.array([self.rho_default for _ in range(self.n_category)])

    def code_activation(self, train_vector):
        activations = []
        for category in range(self.n_category):
            activations.append(np.sum([self.contribution_param * fuzz_min_sum(train_vector[ch], self.w[category][ch])
                                       / (self.gamma + sum(self.w[category][ch])) for ch in range(self.num_channel)]))
        return np.array(activations)

    def readout(self, idx):
        assert idx < self.n_category, "index out of bound"

        return np.array(self.w[idx])

    def delete_category(self, indices, idx):
        for index in indices:
            if index < idx:
                idx -= 1
            self.n_category -= 1
            del self.w[index]
            del self.rho[index]
        return idx

    def _template_matching(self, sample, max_id):
        return np.array([fuzz_min_sum(sample[ch], self.w[max_id][ch]) / (len(sample[ch]) / 2)
                         > self.rho[max_id] for ch in range(self.num_channel)])

    def _complement_coding(self, sample):
        if self.complement_coding:
            result = np.array([np.hstack((sample[ch], 1 - sample[ch])) for ch in range(self.num_channel)])
            return result
        else:
            return sample

    def train(self, x, shuffle=True, train=True):
        classes = []
        for sample in x:
            # phase 1: complement coding
            comp_sample = self._complement_coding(sample)

            # phase 2: code activation
            activations = self.code_activation(comp_sample)
            # phase 3: code competition
            max_id = np.argmax(activations)

            if train:
                # phase 4: template matching
                match_val = self._template_matching(comp_sample, max_id)

                # phase 5: template learning
                if all(match_val):  # template learning occurs
                    for ch in range(self.num_channel):
                        self.w[max_id][ch] = (1 - self.alpha) * self.w[max_id][ch] + self.alpha * np.minimum(
                            comp_sample[ch], self.w[max_id][ch])
                else:  # create a new category
                    self.n_category += 1
                    self.w.append(comp_sample)
                    self.rho = np.append(self.rho, self.rho_default)
                    max_id = self.n_category - 1

            # set last retrieved memory
            self.last_retrieved_memory = max_id
            classes.append(max_id)
        return np.array(classes)

    def test(self, x):
        clustering_result = self.train(x, shuffle=False, train=False)
        return clustering_result


if __name__ == "__main__":
    # Demo: FusionART for clustering
    from numpy import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from utils import make_cluster_data, synthetic_data

    random.seed(43)

    """ 
    To test stability of ART
    1) when the input list is not shuffled: Stable learning
    2) when the input list is shuffled: Unstable learning -> need to increase rho a little bit 
    """

    # 0: limitation of fuzzy AND
    # 1: synthetic data (triangular data)
    # 2: dispersed data (Gaussian distribution)
    example = 2

    if example == 1:
        # Synthetic data 1
        testART = FusionART(1, [2], complement_coding=True, rho=0.5, alpha=1)
        x, y = synthetic_data()

        for i in range(len(x)):
            print(i)
            tmp_class = testART.learn_or_classify([np.array([x[i], y[i]])], learn=True)
            print(testART.weight)

        classified_x, classified_y = [[np.array([]) for _ in range(testART.n_category)] for _ in range(2)]
        for i in range(len(x)):
            tmp_class = testART.learn_or_classify([np.array([x[i], y[i]])], learn=False)
            classified_x[tmp_class] = np.append(classified_x[tmp_class], np.array([x[i]]))
            classified_y[tmp_class] = np.append(classified_y[tmp_class], np.array([y[i]]))

        fig = plt.figure(1)
        ax = fig.add_subplot(111, aspect='equal')

        for i in range(testART.n_category):
            ax.plot(classified_x[i], classified_y[i], 'x')

        for l in testART.weight:
            w11, w12, w21, w22 = l[0][0], l[0][1], l[0][2], l[0][3]
            ax.add_patch(
                patches.Rectangle(
                    (w11, w12),
                    1 - w21 - w11,
                    1 - w22 - w12,
                    fill=False,  # remove background
                    edgecolor='r'
                )
            )
        plt.title('Classification result')
        plt.axis([0, 1, 0, 1])

        plt.show()

    elif example == 2:
        # Synthetic data 2
        testART = FusionART(2, [1, 1], complement_coding=True, rho=0.75)

        # training the FusionART
        x, y = make_cluster_data()
        z = list(zip(x, y))
        random.shuffle(z)
        x, y = zip(*z)

        s_raw_data = []
        for i in range(len(x)):
            # s_raw_data.append([[x[i], y[i]]])
            s_raw_data.append([[x[i]], [y[i]]])
        s_data = np.array(s_raw_data)

        testART.train(s_data, shuffle=True)
        category = testART.test(s_data)

        data_classified_x, data_classified_y = [[np.array([]) for _ in range(testART.n_category)] for _ in range(2)]

        for i in range(len(category)):
            # data_classified_x[int(category[i])] = np.append(data_classified_x[int(category[i])], np.array([s_data[i][0][0]]))
            # data_classified_y[int(category[i])] = np.append(data_classified_y[int(category[i])], np.array([s_data[i][0][1]]))
            data_classified_x[int(category[i])] = np.append(data_classified_x[int(category[i])],
                                                            np.array([s_data[i][0][0]]))
            data_classified_y[int(category[i])] = np.append(data_classified_y[int(category[i])],
                                                            np.array([s_data[i][1][0]]))
        plt.figure()
        for i in range(testART.n_category):
            plt.plot(data_classified_x[i], data_classified_y[i], 'x')

        plt.title('Classification result'), plt.show()

    else:
        art = FusionART(1, [2], complement_coding=True, rho=0.5, alpha=1)
        art.learn_or_classify([np.array([0, 0])])
