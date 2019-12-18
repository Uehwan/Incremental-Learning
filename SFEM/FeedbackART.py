import numpy as np
from EMART import ExpandableInputART
from utils import fuzz_min_sum


class FeedbackART(ExpandableInputART):
    """
    FeedbackART accepts user feedback and modulates the strength of the corresponding
    memory component
    """

    def __init__(self, num_channel, input_dim, complement_coded=True, gamma=0.01,
                 alpha=0.3, rho=0.9, contribution_param="", memory_strength=0.75,
                 memory_decay_factor=0.01, memory_reinforcement=0.15, memory_threshold=0.1, memory=True):
        assert type(input_dim) == list, "input_dim should be a list!"

        super().__init__(num_channel, input_dim, complement_coded, gamma, alpha,
                         rho, contribution_param)

        self.memory = memory  # True => conduct memory evolution
        self.memory_strength_default = memory_strength
        self.memory_decay_factor = memory_decay_factor
        self.memory_reinforcement = memory_reinforcement
        self.memory_threshold = memory_threshold
        self.last_retrieved_memory = 0
        self.rho_decay_factor = self.memory_decay_factor * 2
        self.rho_reinforcement = self.memory_reinforcement / 2

        # memory strength for each category
        self.memory_strength = [self.memory_strength_default for _ in range(self.n_category)]

    def train(self, x, shuffle=True, train=True):
        """
        Employs proposed similarity measure
        """
        classes = []
        for sample in x:
            new_category_generated = False

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
                    for i in range(self.num_channel):
                        self.w[max_id][i] = (1 - self.alpha) * self.w[max_id][i] + \
                                            self.alpha * np.minimum(comp_sample[i], self.w[max_id][i])

                else:  # create a new category
                    self.n_category += 1
                    self.w.append(comp_sample)
                    self.rho = np.append(self.rho, self.rho_default)
                    max_id = self.n_category - 1
                    new_category_generated = True

            # evolve memory strength
            if self.memory:
                max_id, deleted = self.evolve_memory(max_id, new_category_generated)
                if deleted:
                    print("Deleted memory indices: ", deleted)

            # set last retrieved memory
            self.last_retrieved_memory = max_id
            classes.append(max_id)
        return np.array(classes)

    def test(self, x):
        clustering_result = self.train(x, shuffle=False, train=False)
        return clustering_result

    def code_activation(self, train_vector):
        activations = []
        for category in range(self.n_category):
            if np.isscalar(self.rho):
                print("abc")
            if np.isscalar(self.w):
                print("def")
            if np.isscalar(train_vector):
                print("eee")
            if np.isscalar(self.w[category]):
                print("def")

            activations.append(sum([self.contribution_param * fuzz_min_sum(train_vector[i], self.w[category][i])
                                    * self.rho_default / (self.gamma + sum(self.w[category][i]))
                                    / self.rho[category] for i in range(self.num_channel)]))
        return np.array(activations)

    def evolve_memory(self, idx, new_category_generated):
        for ms in range(len(self.memory_strength)):
            self.memory_strength[ms] *= (1 - self.memory_decay_factor)

        if new_category_generated:  # idx == len(self.memory_strength):
            # new category detected
            self.memory_strength.append(self.memory_strength_default)
        else:
            # read the memory strength to process
            tmp = self.memory_strength[idx]
            # recover from decaying
            tmp *= 1 / (1 - self.memory_decay_factor)
            # reinforce the memory
            self.memory_strength[idx] = tmp + (1 - tmp) * self.memory_reinforcement

        # delete categories diminished under the threshold
        to_delete = [category for category, elem in enumerate(self.memory_strength) if elem <= self.memory_threshold]

        new_idx = self.delete_category(to_delete, idx)

        return new_idx, to_delete

    def delete_category(self, indices, idx):
        for index in indices:
            if index < idx:
                idx -= 1
            self.n_category -= 1
            self.w = np.delete(self.w, index, axis=0)
            self.rho = np.delete(self.rho, index, axis=0)
            self.memory_strength = np.delete(self.memory_strength, index, axis=0)

            # del self.w[index]
            # del self.rho[index]
            # del self.memory_strength[index]
        return idx

    def feedback(self, user_feedback):
        # user_feedback: 1 for strong positive, 0 for positive, -1 for negative
        tag = 0  # indicates if a sequence is going to be deleted => need to learn a new sequence if -1
        prev_memory = self.memory_strength[self.last_retrieved_memory]

        # for strong positive feedback
        if user_feedback > 0:
            # curr_memory has been retrieved lastly => just need to add one more reinforcement component
            curr_memory = min(1.0, prev_memory + (1 - prev_memory) * self.memory_reinforcement)
            self.rho[self.last_retrieved_memory] *= (1 - self.rho_decay_factor)

        # for negative feedback
        elif user_feedback < 0:
            # calculate criterion: if the seq was retrieved during the last 10 retrievals
            criterion = prev_memory >= self.memory_strength_default * (1 - self.memory_decay_factor) ** 5
            if criterion:
                # revert reinforcement of prev_memory
                prev_memory = (prev_memory - self.memory_reinforcement) / (1 - self.memory_reinforcement)
                # calculate current memory
                curr_memory = prev_memory * (1 - self.memory_decay_factor) ** 2
            else:
                # need to learn a new sequence
                curr_memory = self.memory_threshold / (1 - self.memory_decay_factor)
                tag = -1

            # increase rho => less probability of activation
            curr_rho = self.rho[self.last_retrieved_memory]
            curr_rho = curr_rho + (1 - curr_rho) * self.rho_reinforcement
            self.rho[self.last_retrieved_memory] = curr_rho

        self.memory_strength[self.last_retrieved_memory] = curr_memory
        return tag


if __name__ == "__main__":
    # Demo: FusionART for clustering
    from numpy import random
    import matplotlib.pyplot as plt
    from utils import make_cluster_data

    random.seed(43)

    testART = FeedbackART(1, [2, ], complement_coded=True, rho=0.9)

    # training the FusionART
    x, y = make_cluster_data()
    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)

    s_raw_data = []
    for i in range(len(x)):
        s_raw_data.append([[x[i], y[i]]])
        # s_raw_data.append([[x[i]], [y[i]]])
    s_data = np.array(s_raw_data)
    """ 
    To test stability of ART
    1) when the input list is not shuffled: Stable learning
    2) when the input list is shuffled: Unstable learning -> need to increase rho a little bit 
    """

    testART.train(s_data, shuffle=True)
    category = testART.test(s_data)

    data_classified_x, data_classified_y = [[np.array([]) for _ in range(testART.n_category)] for _ in range(2)]

    for i in range(len(category)):
        data_classified_x[int(category[i])] = np.append(data_classified_x[int(category[i])],
                                                        np.array([s_data[i][0][0]]))
        data_classified_y[int(category[i])] = np.append(data_classified_y[int(category[i])],
                                                        np.array([s_data[i][0][1]]))
        # data_classified_x[int(category[i])] = np.append(data_classified_x[int(category[i])], np.array([s_data[i][0][0]]))
        # data_classified_y[int(category[i])] = np.append(data_classified_y[int(category[i])], np.array([s_data[i][1][0]]))
    plt.figure()
    for i in range(testART.n_category):
        plt.plot(data_classified_x[i], data_classified_y[i], 'x')

    plt.title('Classification result'), plt.show()
