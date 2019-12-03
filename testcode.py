import numpy as np
from functools import partial
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score
from sklearn import mixture
from sklearn.cluster import KMeans
from drn import DRN
from sdrn import sDRN

import random
import matplotlib.pyplot as plt
import scipy.io as io

import warnings
l2_norm = partial(np.linalg.norm, ord=2, axis=-1)
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':

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

    # Preparing lists
    s_list, r_list = [], []

    # Synthetic data
    # s_data0
    x, y = make_cluster_data()
    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)

    s_raw_data0 = []
    for i in range(len(x)):
        s_raw_data0.append([[x[i], y[i]]])
    s_data0 = np.array(s_raw_data0)
    s_list.append({'data': s_data0})

    # s data1
    s_raw_data1 = io.loadmat('data/2D_data/2D_manual_300.mat')['points']
    s_data1 = np.zeros([300, 1, 2])
    for i in range(300):
        for dim in range(2):
            s_data1[i][0][dim] = s_raw_data1[i][dim]
    #    s_data1 = np.zeros([300,2,1])
    #    for i in range(300):
    #        for ch in range(2):
    #            s_data1[i][ch][0] = s_raw_data1[i][ch]
    s_list.append({'data': s_data1})

    # s data2
    import csv
    s_raw_data2 = []

    with open("data/2D_data/2D_joensuu_2000.CSV") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            s_raw_data2.append(row)
    # s_raw_data2 = np.array(s_raw_data2)
    s_raw_data2 = np.array(s_raw_data2) / 100
    # s_raw_data2 = s_raw_data2 *100000000
    s_data2 = np.expand_dims(s_raw_data2, axis=1)  # channel 1 input_dims = [2,]
    # results = np.expand_dims(results, axis=2) # channel 2 input_dims = [1,1]
    s_list.append({'data': s_data2})
    # Real data
    # Real data0
    r_raw_data0 = []
    f0 = open("data/uci_real_data/lenses_dataset.data", 'r')
    lines0 = f0.readlines()
    for line0 in lines0:
        a = [int(float(s)) for s in line0.split()]
        r_raw_data0.append(a)
    r_raw_data0 = np.array(r_raw_data0)
    r_raw_data0 = np.random.permutation(r_raw_data0)

    r_data0_class = np.array(r_raw_data0[:, 5])
    r_raw_data0 = np.array(r_raw_data0[:, 1:5])

    r_data0 = np.expand_dims(r_raw_data0, axis=1)  # channel 1 input_dims = [4,]
    # r_data0 = np.expand_dims(r_raw_data0, axis=2) # channel 4 input_dims = [1,1,1,1]
    r_list.append({'name': 'lens', 'data': r_data0, 'class': r_data0_class, 'raw_data': r_raw_data0})

    # Real data1
    cache_d1 = []
    f1 = open("data/uci_real_data/balance_scale_dataset.data", 'r')
    lines1 = f1.readlines()
    for line1 in lines1:
        temp = line1.split(',')
        if temp[0] == 'L':
            temp[0] = 1
        elif temp[0] == 'R':
            temp[0] = 2
        elif temp[0] == 'B':
            temp[0] = 3
        a = []
        for s in temp:
            a.append(int(float(s)))
        #        del a[-1]
        # line = line.split(' ')
        cache_d1.append(a)

    shff_cache_d1 = np.random.permutation(cache_d1)

    r_raw_data1 = np.array(shff_cache_d1[:, 1:])
    r_data1_class = np.array(shff_cache_d1[:, 0])

    r_data1 = np.expand_dims(r_raw_data1, axis=1)  # channel 1 input_dims = [4,]
    # r_data1 = np.expand_dims(r_raw_data1, axis=2) # channel 4 input_dims = [1,1,1,1]
    r_list.append({'name': 'balance_scale', 'data': r_data1, 'class': r_data1_class, 'raw_data': r_raw_data1})

    # Real data2
    cache_d2 = []
    f2 = open("data/uci_real_data/bupa_dataset.data", 'r')
    lines2 = f2.readlines()
    for line2 in lines2:
        a = [int(float(s)) for s in line2.split(',')]
        cache_d2.append(a)

    shff_cache_d2 = np.random.permutation(cache_d2)
    r_raw_data2 = np.array(shff_cache_d2[:, :5])
    r_data2_class = np.array(shff_cache_d2[:, 5])

    r_data2 = np.expand_dims(r_raw_data2, axis=1)  # channel 1 input_dims = [5,]
    # r_data2 = np.expand_dims(r_raw_data2, axis=2) # channel 5 input_dims = [1,1,1,1,1]
    r_list.append({'name': 'bupa', 'data': r_data2, 'class': r_data2_class, 'raw_data': r_raw_data2})

    # Real data3
    cache_d3 = []
    f3 = open("data/uci_real_data/transfusion_dataset.data", 'r')
    lines3 = f3.readlines()
    for line3 in lines3:
        a = [int(float(s)) for s in line3.split(',')]
        # del a[-1]
        cache_d3.append(a)

    shff_cache_d3 = np.random.permutation(cache_d3)

    r_raw_data3 = np.array(shff_cache_d3[:, :4])
    r_data3_class = np.array(shff_cache_d3[:, 4])

    r_data3 = np.expand_dims(r_raw_data3, axis=1)  # channel 1 input_dims = [4,]
    # r_data3 = np.expand_dims(r_raw_data3, axis=2) # channel 4 input_dims = [1,1,1,1]
    r_list.append({'name': 'transfusion', 'data': r_data3, 'class': r_data3_class, 'raw_data': r_raw_data3})

    # Real data4
    cache_d4 = []
    f4 = open("data/uci_real_data/data_banknote_authentication.txt", 'r')
    lines4 = f4.readlines()
    for line4 in lines4:
        a = [int(float(s)) for s in line4.split(',')]
        # del a[-1]
        cache_d4.append(a)

    shff_cache_d4 = np.random.permutation(cache_d4)

    r_raw_data4 = np.array(shff_cache_d4[:, :4])
    r_data4_class = np.array(shff_cache_d4[:, 4])

    r_data4 = np.expand_dims(r_raw_data4, axis=1)  # channel 1 input_dims = [4,]
    # r_data4 = np.expand_dims(r_raw_data4, axis=2) # channel 4 input_dims = [1,1,1,1]
    r_list.append({'name': 'banknote_authentication', 'data': r_data4, 'class': r_data4_class, 'raw_data': r_raw_data4})

    # Real data5
    cache_d5 = []
    f5 = open("data/uci_real_data/car.txt", 'r')
    lines5 = f5.readlines()
    for line5 in lines5:
        a = [int(float(s)) for s in line5.split(',')]
        # del a[-1]
        cache_d5.append(a)

    shff_cache_d5 = np.random.permutation(cache_d5)

    r_raw_data5 = np.array(shff_cache_d5[:, :6])
    r_data5_class = np.array(shff_cache_d5[:, 6])

    r_data5 = np.expand_dims(r_raw_data5, axis=1)  # channel 1 input_dims = [6,]
    # r_data5 = np.expand_dims(r_raw_data5, axis=2) # channel 6 input_dims = [1,1,1,1,1,1]
    r_list.append({'name': 'car', 'data': r_data5, 'class': r_data5_class, 'raw_data': r_raw_data5})

    # Real data6
    cache_d6 = []
    f6 = open("data/uci_real_data/Wholesale customers data.csv", 'r')
    lines6 = f6.readlines()
    for line6 in lines6:
        a = [int(float(s)) for s in line6.split(',')]
        # del a[-1]
        cache_d6.append(a)

    shff_cache_d6 = np.random.permutation(cache_d6)

    r_raw_data6 = np.array(shff_cache_d6[:, 2:])
    r_data6_class = 3*np.array(shff_cache_d6[:, 0]) + np.array(shff_cache_d6[:, 1])

    r_data6 = np.expand_dims(r_raw_data6, axis=1)  # channel 1 input_dims = [6,]
    # r_data6 = np.expand_dims(r_raw_data6, axis=2) # channel 6 input_dims = [1,1,1,1,1,1]
    r_list.append({'name': 'wholesale', 'data': r_data6, 'class': r_data6_class, 'raw_data': r_raw_data6})

    # Real data7
    cache_d7 = []
    f7 = open("data/uci_real_data/uci knowledge modeling dataset.CSV", 'r')
    lines7 = f7.readlines()
    for line7 in lines7:
        a = [int(float(s)) for s in line7.split(',')]
        # del a[-1]
        cache_d7.append(a)

    shff_cache_d7 = np.random.permutation(cache_d7)

    r_raw_data7 = np.array(shff_cache_d7[:, :5])
    r_data7_class = np.array(shff_cache_d7[:, 5])

    r_data7 = np.expand_dims(r_raw_data7, axis=1)  # channel 1 input_dims = [6,]
    # r_data7 = np.expand_dims(r_raw_data7, axis=2) # channel 6 input_dims = [1,1,1,1,1,1]
    r_list.append({'name': 'knowledge_modeling', 'data': r_data7, 'class': r_data7_class, 'raw_data': r_raw_data7})


#    elem_val = True; rho_val = 0.7; gp_val = 1; iov_val = 0.5
    elem_val = False; rho_val = 0.5; gp_val = 1; iov_val = 0.5

    s_data0_net = DRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.9, rho=0.9, v=2)
    s_data1_net = DRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.8, rho=0.9, v=2)
    s_data2_net = DRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.8, rho=0.9, v=2)
#    s_data0_net = new_rDRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.9, rho=0.8, v=2, gp=gp_val)
#    s_data1_net = new_rDRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.8, rho=0.9, v=2, gp=gp_val)
#    s_data2_net = new_rDRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.8, rho=0.9, v=2, gp=gp_val)
    r_data0_net = sDRN(num_channel=1, input_dim=[4, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2, gp=gp_val, iov=iov_val)
    r_data1_net = sDRN(num_channel=1, input_dim=[4, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2, gp=gp_val, iov=iov_val)
    r_data2_net = sDRN(num_channel=1, input_dim=[5, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2, gp=gp_val, iov=iov_val)
    r_data3_net = sDRN(num_channel=1, input_dim=[4, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2, gp=gp_val, iov=iov_val)
    r_data4_net = sDRN(num_channel=1, input_dim=[4, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2, gp=gp_val, iov=iov_val)
    r_data5_net = sDRN(num_channel=1, input_dim=[6, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2, gp=gp_val, iov=iov_val)
    r_data6_net = sDRN(num_channel=1, input_dim=[6, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2, gp=gp_val, iov=iov_val)
    r_data7_net = sDRN(num_channel=1, input_dim=[5, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2, gp=gp_val, iov=iov_val)
    # r_data0_net = DRN(num_channel=1, input_dim=[4, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2)
    # r_data1_net = DRN(num_channel=1, input_dim=[4, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2)
    # r_data2_net = DRN(num_channel=1, input_dim=[5, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2)
    # r_data3_net = DRN(num_channel=1, input_dim=[4, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2)
    # r_data4_net = DRN(num_channel=1, input_dim=[4, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2)
    # r_data5_net = DRN(num_channel=1, input_dim=[6, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2)
    # r_data6_net = DRN(num_channel=1, input_dim=[6, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2)
    # r_data7_net = DRN(num_channel=1, input_dim=[5, ], tmp_mat_elem=elem_val, lr=0.8, rho=rho_val, v=2)

    # Add networks to list
    s_list[0]['net'] = s_data0_net; s_list[1]['net'] = s_data1_net; s_list[2]['net'] = s_data2_net
    r_list[0]['net'] = r_data0_net; r_list[1]['net'] = r_data1_net; r_list[2]['net'] = r_data2_net; r_list[3]['net'] = r_data3_net
    r_list[4]['net'] = r_data4_net; r_list[5]['net'] = r_data5_net; r_list[6]['net'] = r_data6_net; r_list[7]['net'] = r_data7_net

    Train_s_list = [0, 1, 2]
    Train_r_list = [1,2,3,4,5,6]
    # Batch train/test split ratio. Higher Ratio: Bigger train, smaller test batch set.
    batch_split_ratio = 0.5

    s_study_list = [s_list[i] for i in Train_s_list]
    r_study_list = [r_list[i] for i in Train_r_list]

    # Batch data train/test for batch learning (KNN, GMM)
    for data_i in range(len(r_study_list)):
        split_edge = int(batch_split_ratio*len(r_study_list[data_i]['raw_data']))
        r_study_list[data_i]['batch_train'] = r_study_list[data_i]['raw_data'][:split_edge]
        r_study_list[data_i]['batch_test'] = r_study_list[data_i]['raw_data'][split_edge:]
        r_study_list[data_i]['batch_test_class'] = r_study_list[data_i]['class'][split_edge:]

    # Train networks
    for data_i in range(len(s_study_list)):
        s_study_list[data_i]['net'].train(s_study_list[data_i]['data'], shuffle=True)
        s_study_list[data_i]['category'] = s_study_list[data_i]['net'].test(s_study_list[data_i]['data'])
    for data_i in range(len(r_study_list)):
        r_study_list[data_i]['net'].train(r_study_list[data_i]['data'], shuffle=True)
        r_study_list[data_i]['category'] = r_study_list[data_i]['net'].test(r_study_list[data_i]['data'])

        r_study_list[data_i]['KNN_net'] = KMeans(n_clusters=r_study_list[data_i]['net'].input_dim[0]).fit(r_study_list[data_i]['batch_train'])
        r_study_list[data_i]['KNN_category'] = r_study_list[data_i]['KNN_net'].predict(r_study_list[data_i]['batch_test'])
        r_study_list[data_i]['GMM_net'] = mixture.GaussianMixture(n_components=r_study_list[data_i]['net'].input_dim[0], covariance_type='full').fit(r_study_list[data_i]['batch_train'])
        r_study_list[data_i]['GMM_category'] = r_study_list[data_i]['GMM_net'].predict(r_study_list[data_i]['batch_test'])

    # Plot 2D(synthetic) result
    for data_i in range(len(s_study_list)):
        data_classified_x, data_classified_y = [[np.array([]) for _ in range(s_study_list[data_i]['net'].n_category)] for _ in range(2)]
        for i in range(len(s_study_list[data_i]['category'])):
            data_classified_x[int(s_study_list[data_i]['category'][i])] = np.append(data_classified_x[int(s_study_list[data_i]['category'][i])],
                                                                     np.array([s_study_list[data_i]['data'][i][0][0]]))
            data_classified_y[int(s_study_list[data_i]['category'][i])] = np.append(data_classified_y[int(s_study_list[data_i]['category'][i])],
                                                                     np.array([s_study_list[data_i]['data'][i][0][1]]))
        plt.figure()
        for i in range(s_study_list[data_i]['net'].n_category):
            plt.plot(data_classified_x[i], data_classified_y[i], 'x')
            plt.gca().add_patch(
                plt.Rectangle((s_study_list[data_i]['net'].w[i][0][0], s_study_list[data_i]['net'].w[i][0][1]),
                              s_study_list[data_i]['net'].w[i][0][2] - s_study_list[data_i]['net'].w[i][0][0],
                              s_study_list[data_i]['net'].w[i][0][3] - s_study_list[data_i]['net'].w[i][0][1], fill=False,
                              edgecolor='b', linewidth=3)
            )
        plt.gca().add_patch(
            plt.Rectangle((s_study_list[data_i]['net'].wg[0][0], s_study_list[data_i]['net'].wg[0][1]),
                          s_study_list[data_i]['net'].wg[0][2] - s_study_list[data_i]['net'].wg[0][0],
                          s_study_list[data_i]['net'].wg[0][3] - s_study_list[data_i]['net'].wg[0][1], fill=False,
                          edgecolor='r', linewidth=3)
        )
        plt.title('Classification result'), plt.show()

    # Function implementation for clustering evaluation
    """
    Clustering Evaluation
    """
    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def check_notsingle_category(list):
        if len(list) == 0:
            print("list empty")
            return False
        else:
            check=list[0]
            for i in range(len(list)):
                if check != list[i]:
                    return True
            return False

# 'class' is answer
# 'categories' is prediction

    # Plot real data result
    for data_i in range(len(r_study_list)):
        print(r_study_list[data_i]['name'])
        if check_notsingle_category(r_study_list[data_i]['category']):
            print("DBI is", davies_bouldin_score(r_study_list[data_i]['raw_data'], r_study_list[data_i]['category']))
            print("NMI is", normalized_mutual_info_score(r_study_list[data_i]['class'], r_study_list[data_i]['category']))
            print("CP is", purity_score(r_study_list[data_i]['class'], r_study_list[data_i]['category']))
            print("KNN DBI is", davies_bouldin_score(r_study_list[data_i]['batch_test'], r_study_list[data_i]['KNN_category']))
            print("KNN NMI is", normalized_mutual_info_score(r_study_list[data_i]['batch_test_class'], r_study_list[data_i]['KNN_category']))
            print("KNN CP is", purity_score(r_study_list[data_i]['batch_test_class'], r_study_list[data_i]['KNN_category']))
            print("GMM DBI is", davies_bouldin_score(r_study_list[data_i]['batch_test'], r_study_list[data_i]['GMM_category']))
            print("GMM NMI is", normalized_mutual_info_score(r_study_list[data_i]['batch_test_class'], r_study_list[data_i]['GMM_category']))
            print("GMM CP is", purity_score(r_study_list[data_i]['batch_test_class'], r_study_list[data_i]['GMM_category']))