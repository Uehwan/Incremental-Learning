import random
import warnings
import numpy as np
import scipy.io as io

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score
from sklearn import mixture
from sklearn.cluster import KMeans

from SFART.utils import make_cluster_data, purity_score, check_not_single_category


if __name__ == '__main__':
    import csv

    warnings.simplefilter(action='ignore', category=FutureWarning)
    random.seed(43)

    # s_list: synthetic data list, r_list: real-world data list
    s_list, r_list = [], []

    # Synthetic data
    """
    Data is synthesized or read and appended to s_list in dictionary format
    data: Reshaped raw data into desired format
    """
    # Synthetic data #0: s_data0
    x, y = make_cluster_data()
    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)

    s_raw_data0 = []
    for i in range(len(x)):
        s_raw_data0.append([[x[i], y[i]]])
    s_data0 = np.array(s_raw_data0)
    s_list.append({'data': s_data0})

    # Synthetic data #1: s_data1
    s_raw_data1 = io.loadmat('data/2D_data/2D_manual_300.mat')['points']
    s_data1 = np.zeros([300, 1, 2])
    for i in range(300):
        for dim in range(2):
            s_data1[i][0][dim] = s_raw_data1[i][dim]
    s_list.append({'data': s_data1})

    # Synthetic data #2: s_data2
    s_raw_data2 = []
    with open("data/2D_data/2D_joensuu_2000.CSV") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            s_raw_data2.append(row)

    s_raw_data2 = np.array(s_raw_data2) / 100
    s_data2 = np.expand_dims(s_raw_data2, axis=1)  # channel 1 input_dims = [2,]
    s_list.append({'data': s_data2})

    # Real data
    """
    Data is read and appended to r_list in dictionary format
    name: Name of data
    raw_data: Original raw data read from file
    data: Reshaped raw data into desired format
    class: Label of each data
    """
    # Real data #0
    r_raw_data0 = []
    f0 = open("data/uci_real_data/lenses_dataset.data", 'r')
    lines0 = f0.readlines()
    for line0 in lines0:
        a = [int(float(s)) for s in line0.split()]
        r_raw_data0.append(a)
    r_raw_data0 = np.array(r_raw_data0)
    r_raw_data0 = np.random.permutation(r_raw_data0) # Shuffle the data

    r_data0_class = np.array(r_raw_data0[:, 5])
    r_raw_data0 = np.array(r_raw_data0[:, 1:5])

    r_data0 = np.expand_dims(r_raw_data0, axis=1)  # channel 1 input_dims = [4,]
    r_list.append({'name': 'lens', 'data': r_data0, 'class': r_data0_class, 'raw_data': r_raw_data0})

    # Real data #1
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
        cache_d1.append(a)

    shff_cache_d1 = np.random.permutation(cache_d1) # Shuffle the data
    r_raw_data1 = np.array(shff_cache_d1[:, 1:])
    r_data1_class = np.array(shff_cache_d1[:, 0])
    r_data1 = np.expand_dims(r_raw_data1, axis=1)  # channel 1 input_dims = [4,]
    r_list.append({'name': 'balance_scale', 'data': r_data1, 'class': r_data1_class, 'raw_data': r_raw_data1})

    # Real data #2~7
    db_names = ['bupa', 'transfusion', 'banknote_authentication', 'car', 'wholesale', 'knowledge_modeling']
    file_names = ["data/uci_real_data/bupa_dataset.data", 
                  "data/uci_real_data/transfusion_dataset.data", 
                  "data/uci_real_data/data_banknote_authentication.txt",
                  "data/uci_real_data/car.txt",
                  "data/uci_real_data/Wholesale customers data.csv",
                  "data/uci_real_data/uci knowledge modeling dataset.CSV"
                  ]
    data_ranges = [[0,5], [0,4], [0,4], [0,6], [2,100], [0,5]]

    for db_name, file, one_range in zip(db_names, file_names, data_ranges):
        cache = []
        f = open(file, 'r')
        lines = f.readlines()
        for line in lines:
            a = [int(float(s)) for s in line.split(',')]
            cache.append(a)

        shff_cache = np.random.permutation(cache) # Shuffle the data
        r_raw_data = np.array(shff_cache[:, one_range[0]:one_range[1]])
        if db_name =='wholesale':
            r_data_class = 3*np.array(shff_cache[:, 0]) + np.array(shff_cache[:, 1])
        else:
            r_data_class = np.array(shff_cache[:, one_range[1]])

        r_data = np.expand_dims(r_raw_data, axis=1)
        r_list.append({'name': db_name, 'data': r_data, 'class': r_data_class, 'raw_data': r_raw_data})

    # Scaling real dataset size
    for i in range(7):
        r_list[i]['data'] = r_list[i]['data'] * 1
        r_list[i]['raw_data'] = r_list[i]['raw_data'] * 1

    # Set parameters
    batch_split_ratio = 0.5 # Ratio between train / test set split
    elem_val = False; rho_val = 0.7; gp_val = 1; iov_val = 0.5 # Occupied drn/sdrn parameters
    Train_r_list = [1, 2, 3, 4, 5, 6] # Select desired datasets to experiment
    num = 100 # Number of iterations for estimating mean and variation of results

    # Define dictionary to append simulation results
    knn_results = {}; gmm_results = {}
    for data_i in range(len(Train_r_list)):
        knn_results[data_i] = {'DBI': [], 'NMI': [], 'CP': [], 'name': []}
        gmm_results[data_i] = {'DBI': [], 'NMI': [], 'CP': [], 'name': []}

    for i in range(num):
        # Ground-truth cluster numbers
        r_list[0]['ch'] = 3; r_list[1]['ch'] = 3; r_list[2]['ch'] = 2; r_list[3]['ch'] = 2; r_list[4]['ch'] = 2;
        r_list[5]['ch'] = 4; r_list[6]['ch'] = 6; r_list[7]['ch'] = 4

        r_study_list = [r_list[i] for i in Train_r_list]

        # Split dataset to batch_train and batch_test
        for data_i in range(len(r_study_list)):
            split_edge = int(batch_split_ratio*len(r_study_list[data_i]['raw_data']))
            r_study_list[data_i]['batch_train'] = r_study_list[data_i]['raw_data'][:split_edge]
            r_study_list[data_i]['batch_test'] = r_study_list[data_i]['raw_data'][split_edge:]
            r_study_list[data_i]['batch_test_class'] = r_study_list[data_i]['class'][split_edge:]

        # Train KNN, GMM with batch_train
        # Attain category results with batch_test
        for data_i in range(len(r_study_list)):
            r_study_list[data_i]['KNN_net'] = KMeans(n_clusters=r_study_list[data_i]['ch']).fit(r_study_list[data_i]['batch_train'])
            r_study_list[data_i]['KNN_category'] = r_study_list[data_i]['KNN_net'].predict(r_study_list[data_i]['batch_test'])
            r_study_list[data_i]['GMM_net'] = mixture.GaussianMixture(n_components=r_study_list[data_i]['ch'], covariance_type='full').fit(r_study_list[data_i]['batch_train'])
            r_study_list[data_i]['GMM_category'] = r_study_list[data_i]['GMM_net'].predict(r_study_list[data_i]['batch_test'])

        # Evaluate DBI, NMI, CP results for each KNN/GMM category results
        for data_i in range(len(r_study_list)):
            # print(r_study_list[data_i]['name'])
            if check_not_single_category(r_study_list[data_i]['KNN_category']):
                knn_results[data_i]['name'].append(r_study_list[data_i]['name'])
                knn_results[data_i]['DBI'].append(davies_bouldin_score(r_study_list[data_i]['batch_test'], r_study_list[data_i]['KNN_category']))
                knn_results[data_i]['NMI'].append(normalized_mutual_info_score(r_study_list[data_i]['batch_test_class'], r_study_list[data_i]['KNN_category']))
                knn_results[data_i]['CP'].append(purity_score(r_study_list[data_i]['batch_test_class'], r_study_list[data_i]['KNN_category']))

            if check_not_single_category(r_study_list[data_i]['GMM_category']):
                gmm_results[data_i]['name'].append(r_study_list[data_i]['name'])
                gmm_results[data_i]['DBI'].append(davies_bouldin_score(r_study_list[data_i]['batch_test'], r_study_list[data_i]['GMM_category']))
                gmm_results[data_i]['NMI'].append(normalized_mutual_info_score(r_study_list[data_i]['batch_test_class'], r_study_list[data_i]['GMM_category']))
                gmm_results[data_i]['CP'].append(purity_score(r_study_list[data_i]['batch_test_class'], r_study_list[data_i]['GMM_category']))

    # Print the final results
    for data_i in range(len(r_study_list)):
        print('knn',knn_results[data_i]['name'][0])
        print('DBI mean:', np.mean(knn_results[data_i]['DBI']), 'DBI std:', np.std(knn_results[data_i]['DBI']))
        print('NMI mean:', np.mean(knn_results[data_i]['NMI']), 'NMI std:', np.std(knn_results[data_i]['NMI']))
        print('CP mean:', np.mean(knn_results[data_i]['CP']), 'CP std:', np.std(knn_results[data_i]['CP']))
        print('gmm',gmm_results[data_i]['name'][0])
        print('DBI mean:', np.mean(gmm_results[data_i]['DBI']), 'DBI std:', np.std(gmm_results[data_i]['DBI']))
        print('NMI mean:', np.mean(gmm_results[data_i]['NMI']), 'NMI std:', np.std(gmm_results[data_i]['NMI']))
        print('CP mean:', np.mean(gmm_results[data_i]['CP']), 'CP std:', np.std(gmm_results[data_i]['CP']))
