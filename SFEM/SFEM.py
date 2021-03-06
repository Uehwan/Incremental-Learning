import numpy as np
from FeedbackART import FeedbackART


# TODO: not to use index 0 <= no meaning
class SFEM:
    """
    SF-EM from the paper below
    - Kim, Ue-Hwan, and Jong-Hwan Kim. "A Stabilized Feedback Episodic Memory (SF-EM) and
      Home Service Provision Framework for Robot and IoT Collaboration."
      IEEE transactions on cybernetics (2018).
    """

    def __init__(self, num_channel, input_dim, complement_coded=True, gamma=0.01, alpha1=0.5,
                 alpha2=0.5, rho1=0.77, rho2=0.95, contribution_param="", memory_strength=0.75,
                 memory_decay_factor=0.01, memory_reinforcement=0.15, memory_threshold=0.1):
        # ART 1: input to event (complement coding: O, memory evolution: X)
        self.event_layer = FeedbackART(num_channel, input_dim, complement_coded, gamma, alpha1,
                                       rho1, contribution_param, memory_strength, memory_decay_factor,
                                       memory_reinforcement, memory_threshold, False)
        # ART 2: event to episode (complement coding: X, memory evolution: O)
        self.episode_layer = FeedbackART(1, [self.event_layer.n_category], False, 0, alpha2,
                                         rho2, contribution_param, memory_strength, memory_decay_factor,
                                         memory_reinforcement, memory_threshold, True)

    def train(self, sequence, learn=True):
        # indices = np.array([self.event_layer.train(seq, learn) for seq in sequence])
        indices = np.array(self.event_layer.train(sequence, learn))
        while self.episode_layer.input_dim[0] < self.event_layer.n_category:
            self.episode_layer.increase_input_field()
        encoded_episode = np.array([self.encode_sequence(indices)])
        episode_idx = self.episode_layer.train(encoded_episode, learn)
        return episode_idx

    def test(self, sequence, learn=True):
        clustering_result = self.train(sequence, shuffle=False, train=False)
        return clustering_result

    def readout_event(self, idx):
        assert idx < self.episode_layer.n_category, "index out of bound"
        sequence = self.decode_sequence(self.episode_layer.w[idx])

        return [self.event_layer.readout(seq) for seq in sequence]

    def readout_episode(self, idx):
        assert idx < self.episode_layer.n_category, "index out of bound"

        return self.decode_sequence(self.episode_layer.w[idx])

    @staticmethod
    def decode_sequence(in_vector):
        """ Decode deep-art sequences

        >>> test_vec = [np.array([10, 32, 16, 68, 1]) / 100]
        >>> EMART.decode_sequence(test_vec)
        [3, 1, 2, 0, 3, 0, 4]
        """
        indices = []
        vector = in_vector[0]

        # reverse normalization
        m = max([len(str(v)) for v in vector])  # because of floating point error, str conversion was used
        vector = vector * 10 ** (m - 2)

        while any(vector > 0):
            max_power = max([len("{0:b}".format(int(v))) for v in vector])
            curr_idx = np.argmax(vector)
            indices.append(curr_idx)
            vector[curr_idx] -= 2 ** (max_power - 1)

        return np.array(indices)

    def encode_sequence(self, indices):
        vector = np.zeros(self.event_layer.n_category)

        # Following Deep ART encoding => no need to employ buffer channel in implementation
        for idx in indices:
            tmp = np.zeros(self.event_layer.n_category)
            tmp[idx] = 1
            vector = vector * 2 + tmp

        # normalization
        m = int(np.log10(max(vector)))
        vector = vector / (10 ** (m + 1))

        return np.array([vector])

    def prediction_with_partial_cue(self, sequence, raw=True):
        if raw:
            cue_idx = self.learn_or_classify(sequence, False)
        else:
            encoded_seq = self.encode_sequence(sequence)
            cue_idx = self.episode_layer.learn_or_classify(encoded_seq, False)

        return self.readout_episode(cue_idx)


if __name__ == "__main__":
    from numpy import random
    from utils import make_2d_seq_data

    # Test: 2-D case
    random.seed(43)

    art = SFEM(2, [1, 1])
    num_episode = 10
    min_seq_len = 2
    max_seq_len = 10

    # generate a training set
    episodes = [make_2d_seq_data(random.randint(min_seq_len, max_seq_len)) for _ in range(num_episode)]

    # training the EMART
    for i in range(len(episodes)):
        art.train(episodes[i], learn=True)

    # print out the training result
    for i in range(len(episodes)):
        print("episode", i + 1, ": ", art.readout_episode(i + 1))
