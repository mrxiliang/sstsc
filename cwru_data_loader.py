import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
import math
import numpy
from sklearn.model_selection import train_test_split

import os
import pickle
from utils.helper import get_df_all
from pathlib import Path

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]

def load_bearing_raw(dataset_path, dataset_name):
    df_all = get_df_all(dataset_path, segment_length=1024, normalize=True)
    features = df_all.columns[2:]
    target = 'label'
    x_all = df_all[features].values
    y_all = df_all[target].values
    return x_all, y_all

def split_train(x_train, y_train):
    l_images = []
    l_labels = []
    n_labels_per_cls = 10
    label_idxs = np.unique(y_train)

    for c in label_idxs:
        cls_mask = (y_train == c)
        c_images = x_train[cls_mask]
        c_labels = y_train[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
    l_images = np.concatenate(l_images, 0)
    l_labels = np.concatenate(l_labels, 0)
    return l_images, l_labels


def load_cwru(opt):
    '''

    map_label = {'N':0, 'B':1, 'IR':2, 'OR':3}

    References:
    https://github.com/XiongMeijing/CWRU-1

    Args:
        opt:
        dataset_name:

    Returns:

    '''

    ##################
    # load raw data
    ##################
    dataset_path_ = Path('/repository/Machine/CWRU/12k_DE')
    dataset_path='/repository/Machine'
    dataset_name=opt.dataset_name


    with open('{}/{}/{}_data.pickle'.format(dataset_path, dataset_name, dataset_name), 'rb') as handle1:
        data = pickle.load(handle1)
    with open('{}/{}/{}_label.pickle'.format(dataset_path, dataset_name, dataset_name), 'rb') as handle2:
        label = pickle.load(handle2)

    nb_dims = 1
    nb_class = len(np.unique(label))

    nb_timesteps = int(data.shape[1])
    input_shape = (nb_timesteps, nb_dims)

    ############################################
    # Combine all train and test data for resample
    ############################################

    x_all = data
    y_all = label
    ts_idx = list(range(x_all.shape[0]))

    np.random.seed(opt.seed)
    np.random.shuffle(ts_idx)
    x_all = x_all[ts_idx]
    y_all = y_all[ts_idx]

    # calculate the number of each class
    label_idxs = np.unique(y_all)
    class_stat_all = {}
    for idx in label_idxs:
        class_stat_all[idx] = len(np.where(y_all == idx)[0])

    print("[Stat] All class: {}".format(class_stat_all))


    x_train_, x_test, y_train_, y_test = train_test_split(x_all, y_all, test_size = 0.2, random_state=opt.seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_, y_train_, test_size = 0.2, random_state=opt.seed)


    label_idxs = np.unique(y_train)
    class_stat = {}
    for idx in label_idxs:
        class_stat[idx] = len(np.where(y_train == idx)[0])
    print("[Stat] Train class: mean={}, std={}".format(np.mean(list(class_stat.values())),
                                                       np.std(list(class_stat.values()))))

    label_idxs = np.unique(y_val)
    class_stat = {}
    for idx in label_idxs:
        class_stat[idx] = len(np.where(y_val == idx)[0])
    print("[Stat] Val class: mean={}, std={}".format(np.mean(list(class_stat.values())),
                                                     np.std(list(class_stat.values()))))

    label_idxs = np.unique(y_test)
    class_stat = {}
    for idx in label_idxs:
        class_stat[idx] = len(np.where(y_test == idx)[0])
    print("[Stat] Test class: mean={}, std={}".format(np.mean(list(class_stat.values())),
                                                      np.std(list(class_stat.values()))))

    ########################################
    # Data Split End
    ########################################

    # Process data
    x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
    x_val = x_val.reshape((-1, input_shape[0], input_shape[1]))
    x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

    print("[Stat] All class: train_shape:{}, val_shape:{}, test_shape:{}, class_num={}". \
          format(x_train.shape, x_val.shape, x_test.shape, y_all.shape))

    return x_train, y_train, x_val, y_val, x_test, y_test, nb_class, nb_dims


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dataset_name', type=str, default='mmi20', help='Random seed')


    opt = parser.parse_args()
    load_cwru(opt)


