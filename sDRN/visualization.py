import random
import warnings
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from SFART.utils import make_cluster_data
from drn import DRN
from sdrn import sDRN


if __name__ == '__main__':

    warnings.simplefilter(action='ignore', category=FutureWarning)
    random.seed(43)

    # Preparing lists
    s_list, r_list = [], []

    # Synthetic data
    """
    Data is synthesized or read and appended to s_list in dictionary format
    data: Reshaped raw data into desired format
    """
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
    s_list.append({'data': s_data1})

    # s data2
    import csv
    s_raw_data2 = []

    with open("data/2D_data/2D_joensuu_2000_2_easiest_2.csv") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            s_raw_data2.append([float(row[0]),float(row[1])])
    s_raw_data2 = np.array(s_raw_data2) / 10
    s_data2 = np.expand_dims(s_raw_data2, axis=1)  # channel 1 input_dims = [2,]
    s_list.append({'data': s_data2})

    # Parameters setting for DRN/sDRN
    # elem_val = True; rho_val = 0.7; gp_val = 1; iov_val = 0.5
    # elem_val = False; rho_val = 0.5; gp_val = 1; iov_val = 0.5
    elem_val = False; rho_val = 0.7; gp_val = 0; iov_val = 0.8

#    s_data0_net = DRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.9, rho=0.9, v=2)
#    s_data1_net = DRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.8, rho=0.9, v=2)
#    s_data2_net = DRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.8, rho=0.9, v=2)
    s_data0_net = sDRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.9, rho=rho_val, v=2, gp=gp_val)
    s_data1_net = sDRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.8, rho=rho_val, v=2, gp=gp_val)
    s_data2_net = sDRN(num_channel=1, input_dim=[2, ], tmp_mat_elem=True, lr=0.8, rho=rho_val, v=2, gp=gp_val)

    # Add networks to list
    s_list[0]['net'] = s_data0_net; s_list[1]['net'] = s_data1_net; s_list[2]['net'] = s_data2_net

    # Choose datasets to experiment
    Train_s_list = [0, 1, 2]
    s_study_list = [s_list[i] for i in Train_s_list]

    # Train networks
    for data_i in range(len(s_study_list)):
        s_study_list[data_i]['net'].train(s_study_list[data_i]['data'], shuffle=True)
        s_study_list[data_i]['category'] = s_study_list[data_i]['net'].test(s_study_list[data_i]['data'])

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
            # plt.plot(data_classified_x[i], data_classified_y[i], 'o', ms=3.0)
            plt.plot(data_classified_x[i], data_classified_y[i], 'x')
            plt.gca().add_patch(
                plt.Rectangle((s_study_list[data_i]['net'].w[i][0][0], s_study_list[data_i]['net'].w[i][0][1]),
                              s_study_list[data_i]['net'].w[i][0][2] - s_study_list[data_i]['net'].w[i][0][0],
                              s_study_list[data_i]['net'].w[i][0][3] - s_study_list[data_i]['net'].w[i][0][1], fill=False,
                              edgecolor='b', linewidth=1.5)
            )
        plt.gca().add_patch(
            plt.Rectangle((s_study_list[data_i]['net'].wg[0][0], s_study_list[data_i]['net'].wg[0][1]),
                          s_study_list[data_i]['net'].wg[0][2] - s_study_list[data_i]['net'].wg[0][0],
                          s_study_list[data_i]['net'].wg[0][3] - s_study_list[data_i]['net'].wg[0][1], fill=False,
                          edgecolor='k', linewidth=1.5)
        )
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.title('Classification Results'), plt.show()

    # Plot Input Scale Effect On Clustering Performance
    plt.plot(['x1', 'x10', 'x100', 'x1000', 'x10000', 'x100000'], [0.7008, 0.6603, 0.7075,0.8931,0.921,0.9365], 'o-',color='#D62728', linewidth=2.0, ms=5)
    plt.plot(['x1', 'x10', 'x100', 'x1000', 'x10000', 'x100000'], [0.8825, 1.0603, 2.5699,3.3819,2.6481,3.2976], 'o-',color='#1F77B4', linewidth=2.0, ms=5)
    plt.plot(['x1', 'x10', 'x100', 'x1000', 'x10000', 'x100000'], [0.8286, 1.258, 0.8188,1.2368,0.9979,0.8921], 'o-',color='#2CA02C', linewidth=2.0, ms=5)
    plt.plot(['x1', 'x10', 'x100', 'x1000', 'x10000', 'x100000'], [1.5069, 1.6883, 1.4508,1.7301,1.5871,1.6178], 'o-',color='#FF7F0E', linewidth=2.0, ms=5)
    # plt.plot([2,3,4],[5,6,7])
    plt.xlabel('Input Scale',fontsize=15)
    plt.ylabel('DBI Value',fontsize=15)
    # plt.title('Input Scale Effect On \n Clustering Performance',fontsize=15)
    plt.legend(['s-DRN','DRN','k-means','GMM'])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylim((0.5,3.5))
    plt.show()
    
    # Plot Vigilance Effect On Clustering Performance
    plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.4792,0.4817,0.498,0.4544,0.4852,0.4841,0.4705,0.4833,0.4048], marker='o',color='#D62728', linewidth=2.0, ms=5)
    plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.2322,0.2562,0.4174,0.3653,0.5045,0.5725,0.5737,0.9219,0.9271], marker='o',color='#1F77B4', linewidth=2.0, ms=5)
    plt.xlabel('Vigilance Value',fontsize=15)
    plt.ylabel('DBI Value',fontsize=15)
    # plt.title('Vigilance Effect On \n Clustering Performance',fontsize=15)
    plt.legend(['s-DRN','DRN'])
    plt.ylim((0,1))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()

    # Plot Vigilance Effect On Clustering Performance
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [1.0659, 1.003, 1.0871, 0.9959, 1.0638, 1.0849, 1.0748, 1.0454, 0.8308], marker='o', linewidth=2.0, ms=5)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.5061, 0.4337, 0.4187, 0.5586, 0.7873, 0.7554, 0.8399, 0.9472, 0.9019], marker='o', linewidth=2.0, ms=5)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.4792, 0.4817, 0.498, 0.4544, 0.4852, 0.4841, 0.4705, 0.4833, 0.4048], marker='o', linewidth=2.0, ms=5)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [1.1019, 0.8147, 0.8094, 0.8643, 0.9473, 0.9439, 0.9708, 0.9535, 0.9015], marker='o', linewidth=2.0, ms=5)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [1.5688, 1.4125, 1.1877, 1.2367, 1.1735, 1.1355, 1.1175, 1.0779, 0.8394], marker='o', linewidth=2.0, ms=5)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.1408, 0.2559, 0.3493, 0.3382, 0.2963, 0.6641, 0.5757, 0.7031, 0.8725], marker='o', linewidth=2.0, ms=5)
    
    plt.xlabel('Vigilance Value',fontsize=15)
    plt.ylabel('DBI Value',fontsize=15)
    # plt.title('Vigilance Effect On \n Clustering Performance',fontsize=15)
    plt.legend(['balance scale','liver disorder','blood transfusion', 'banknote authentication', 'car', 'wholesale'],fontsize=9)
    plt.ylim((0,4))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()

    # Plot Vigilance Effect On Clustering Performance
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [1.8305, 1.8455, 1.5626, 1.5076, 1.377, 1.3123, 1.2369, 1.2798, 1.0953], marker='o', linewidth=2.0, ms=5)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.4077, 0.5423, 0.5075, 0.8899, 0.9213, 0.933, 1.1056, 0.8929, 1.1556], marker='o', linewidth=2.0, ms=5)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.2322, 0.2562, 0.4174, 0.3653, 0.5045, 0.5725, 0.5737, 0.9219, 0.9271], marker='o', linewidth=2.0, ms=5)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.7002, 0.6958, 0.8338, 1.377, 0.967, 0.8699, 0.9898, 0.8318, 1.0444], marker='o', linewidth=2.0, ms=5)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [2.092, 1.8268, 1.9337, 1.7701, 0.9213, 1.7681, 1.6426, 1.7589, 1.595], marker='o', linewidth=2.0, ms=5)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [1.1665, 1.9008, 2.8704, 2.8959, 2.336, 2.5514, 2.3004, 2.2068, 2.253], marker='o', linewidth=2.0, ms=5)
    
    plt.xlabel('Vigilance Value',fontsize=15)
    plt.ylabel('DBI Value',fontsize=15)
    # plt.title('Vigilance Effect On \n Clustering Performance',fontsize=15)
    plt.legend(['balance scale','liver disorder','blood transfusion', 'banknote authentication', 'car', 'wholesale'],fontsize=9)
    plt.ylim((0,4))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()