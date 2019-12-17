import random
import warnings
import numpy as np
import scipy.io as io

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score

from SFART.utils import make_cluster_data, purity_score, check_not_single_category
from drn import DRN
from sdrn import sDRN


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
    # s_raw_data2 = np.array(s_raw_data2)
    s_raw_data2 = np.array(s_raw_data2) / 100
    # s_raw_data2 = s_raw_data2 *100000000
    s_data2 = np.expand_dims(s_raw_data2, axis=1)  # channel 1 input_dims = [2,]
    # results = np.expand_dims(results, axis=2) # channel 2 input_dims = [1,1]
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
    r_raw_data0 = np.random.permutation(r_raw_data0)

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

    shff_cache_d1 = np.random.permutation(cache_d1)

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
    data_ranges = [[0, 5], [0, 4], [0, 4], [0, 6], [2, 100], [0, 5]]

    for db_name, file, one_range in zip(db_names, file_names, data_ranges):
        cache = []
        f = open(file, 'r')
        lines = f.readlines()
        for line in lines:
            a = [int(float(s)) for s in line.split(',')]
            cache.append(a)

        shff_cache = np.random.permutation(cache)
        r_raw_data = np.array(shff_cache[:, one_range[0]:one_range[1]])
        if db_name == 'wholesale':
            r_data_class = 3 * np.array(shff_cache[:, 0]) + np.array(shff_cache[:, 1])
        else:
            r_data_class = np.array(shff_cache[:, one_range[1]])

        r_data = np.expand_dims(r_raw_data, axis=1)
        r_list.append({'name': db_name, 'data': r_data, 'class': r_data_class, 'raw_data': r_raw_data})

    # Scaling real dataset size
    for i in range(7):
        r_list[i]['data'] = r_list[i]['data'] * 1
        r_list[i]['raw_data'] = r_list[i]['raw_data'] * 1

    # Set parameters
    # elem_val = False; rho_val = 0.5; gp_val = 1; iov_val = 0.5
    # elem_val = True; rho_val = 0.7; gp_val = 1; iov_val = 0.5
    elem_val = True; rho_val = 0.9; gp_val = 1; iov_val = 0.5  # Occupied drn/sdrn parameters
    num = 100 # Number of iterations for estimating mean and variation of results
    # Train_r_list = [1, 2, 3, 4, 5, 6]
    Train_r_list = [1, 2, 3, 4, 5, 6] # Select desired datasets to experiment

    # Define dictionary to append simulation results
    results = {}
    for data_i in range(len(Train_r_list)):
        results[data_i] = {'DBI': [], 'NMI': [], 'CP': [], 'name': []}

    for i in range(num):
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
        r_list[0]['net'] = r_data0_net; r_list[1]['net'] = r_data1_net; r_list[2]['net'] = r_data2_net; r_list[3]['net'] = r_data3_net
        r_list[4]['net'] = r_data4_net; r_list[5]['net'] = r_data5_net; r_list[6]['net'] = r_data6_net; r_list[7]['net'] = r_data7_net

        r_study_list = [r_list[i] for i in Train_r_list]

        # Train networks
        # Attain category results
        for data_i in range(len(r_study_list)):
            r_study_list[data_i]['net'].train(r_study_list[data_i]['data'], shuffle=True)
            r_study_list[data_i]['category'] = r_study_list[data_i]['net'].test(r_study_list[data_i]['data'])

        # Evaluate DBI, NMI, CP results for category results
        for data_i in range(len(r_study_list)):
            if check_not_single_category(r_study_list[data_i]['category']):
                results[data_i]['name'].append(r_study_list[data_i]['name'])
                results[data_i]['DBI'].append(davies_bouldin_score(r_study_list[data_i]['raw_data'], r_study_list[data_i]['category']))
                results[data_i]['NMI'].append(normalized_mutual_info_score(r_study_list[data_i]['class'], r_study_list[data_i]['category']))
                results[data_i]['CP'].append(purity_score(r_study_list[data_i]['class'], r_study_list[data_i]['category']))

    # Print the final results
    for data_i in range(len(r_study_list)):
        print(results[data_i]['name'])
        print('DBI mean:', np.mean(results[data_i]['DBI']), 'DBI std:', np.std(results[data_i]['DBI']))
        print('NMI mean:', np.mean(results[data_i]['NMI']), 'NMI std:', np.std(results[data_i]['NMI']))
        print('CP mean:', np.mean(results[data_i]['CP']), 'CP std:', np.std(results[data_i]['CP']))
