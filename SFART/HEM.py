import numpy as np
from EMART import EMART


class HEM():
    def __init__(self):
        pass

    # contributions 1. input stablization process
    def learn_or_classify(self, sequence, learn=True):
        pass

    # contributions 2. increased accuracy of retrieval process
    def prediction_with_partial_cue(self, sequence):
        pass

    # contributions 3. user feedback => weight modification (supervised phase)
    def user_feedback(self):
        pass


def seq_pad(seq):
    max_len = max([len(s) for s in seq])
    vec_size = len(seq[0][0])
    PAD_token = np.float32([0 for _ in range(vec_size)])

    for s in seq:
        curr_len = len(s)
        s += [PAD_token for _ in range(max_len - curr_len)]

    return seq


if __name__ == "__main__":
    import csv

    # IoT simulation
    f = open('data.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)

    next(rdr)  # first line: class
    next(rdr)  # second line: ids

    data, episode = [], []

    data_tf = []
    episode_tf = []

    startIdx1, startIdx2, startIdx3, startIdx4 = 2, 17, 35, 51
    endIdx1, endIdx2, endIdx3, endIdx4 = 17, 35, 51, 83

    for l in rdr:
        if not any(l):
            data.append(episode)
            data_tf.append(episode_tf)
            episode = []
            episode_tf = []
            continue

        l1, l2, l3, l4 = l[startIdx1:endIdx1], l[startIdx2:endIdx2], l[startIdx3:endIdx3], l[startIdx4:endIdx4]
        vec = [np.float32(l1), np.float32(l2), np.float32(l3), np.float32(l4)]
        vec_tf = np.float32(l[startIdx1:endIdx4])
        episode.append(vec)
        episode_tf.append(vec_tf)

    data.append(episode) # last episode
    data_tf.append(episode_tf)

    data_tf = seq_pad(data_tf)
    x_data = [d[:-1] for d in data_tf]
    y_data = [d[1:] for d in data_tf]

    f.close()

    art = EMART(4, [15, 18, 16, 32], False, 0.01, 0.5, 0.5, 0.9, 0.95)
    # art = EMART(15+18+16+32, [1 for _ in range(32)], True, 0.01, 0.5, 0.5, 0.9, 0.95)

    for d in data:
        art.learn_or_classify(d)

    total_num_of_services = 0
    total_num_of_missed = 0

    for seq in data:
        full_length = len(seq)
        half_length = round(len(seq)/2)

        true_val = art.prediction_with_partial_cue(seq, True)
        retrieved = art.prediction_with_partial_cue(seq[0:half_length], True)

        matched_length = len(list(set(true_val).intersection(retrieved)))
        total_num_of_services += full_length
        total_num_of_missed += full_length - matched_length

    print("HEM result, error rate:", total_num_of_missed/total_num_of_services, total_num_of_missed, total_num_of_services)

    # (2) Comparison with LSTM
    # LSTM in Tensorflow
    # import tensorflow as tf
    # from tensorflow.contrib import rnn
    ''''
    tf.set_random_seed(7)  # reproducibility
    
    input_dim = endIdx4 - 2
    output_dim = input_dim
    hidden_size = input_dim
    batch_size = 51  # one sentence
    sequence_length = 8
    alpha = 0.1

    X = tf.placeholder(
        tf.float32, [None, sequence_length, input_dim])  # X one-hot
    Y = tf.placeholder(tf.int32, [None, sequence_length, output_dim])  # Y label

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(
        cell, X, initial_state=initial_state, dtype=tf.float32)

    # FC layer
    X_for_fc = tf.reshape(outputs, [-1, hidden_size])
    outputs = tf.contrib.layers.fully_connected(
        inputs=X_for_fc, num_outputs=output_dim, activation_fn=tf.tanh)

    # reshape out for sequence_loss
    outputs = tf.reshape(outputs, [batch_size, sequence_length, output_dim])

    weights = tf.ones([batch_size, sequence_length])
    sequence_loss = tf.losses.mean_squared_error(Y, outputs)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(alpha=alpha).minimize(loss)

    k = sum([tf.reduce_sum(tf.cast(Y[i], tf.float32) - tf.round(outputs[i])) for i in range(batch_size)])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
            if i % 100 == 0:
                tk = sess.run(k, feed_dict={X: x_data, Y: y_data})
                # print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)
                print(i, "loss: ", l, "k: ", tk)
    '''
