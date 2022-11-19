# -*- coding: utf-8 -*-
import utils.helper as hlp
import numpy as np
from dataloader.ucr2018 import load_ucr2018
import utils.augmentation as aug
import os
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from pyts.datasets import load_gunpoint
from pyts.transformation import ShapeletTransform

def plot_shapelet():
    X_train, _, y_train, _ = load_gunpoint(return_X_y=True)

    # Shapelet transformation
    st = ShapeletTransform(window_sizes=[12, 24, 36, 48],
                           random_state=42, sort=True)
    X_new = st.fit_transform(X_train, y_train)

    # Visualize the four most discriminative shapelets
    plt.figure(figsize=(6, 4))
    for i, index in enumerate(st.indices_[:4]):
        idx, start, end = index
        plt.plot(X_train[idx], color='C{}'.format(i),
                 label='Sample {}'.format(idx))
        plt.plot(np.arange(start, end), X_train[idx, start:end],
                 lw=5, color='C{}'.format(i))

    plt.xlabel('Time', fontsize=12)
    plt.title('The four more discriminative shapelets', fontsize=14)
    plt.legend(loc='best', fontsize=8)
    plt.show()

def plot1d(x, x2=None, x3=None,linestyle='-',
           linewidth=4, color='#FF8C00', figsize=(6, 3),
           ylim=(-1, 1), save_file=""):

    plt.figure(figsize=figsize)
    steps = np.arange(x.shape[0])
    plt.plot(steps, x, color=color, linestyle=linestyle, linewidth=linewidth)
    if x2 is not None:
        plt.plot(steps, x2, color='#FF4500', linestyle='--', linewidth=linewidth)
    if x3 is not None:
        plt.plot(steps, x3, 'k--', linewidth=linewidth)
    plt.ylim((np.min(x) - 0.1, np.max(x) + 0.1))
    plt.axis('off')
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        print("save {}!".format(save_file))
    else:
        plt.show()
    plt.close()
    return


def plot_crop(x_org, x_crop1, idx1, figsize=(5, 2), name='x_org'):
    plt.figure(figsize=figsize)
    plt.plot(x_org, color='C0', linestyle='--', lw=4)
    plt.plot(np.arange(idx1[0], idx1[1]), x_crop1,
             lw=8, color='C1', linestyle='-')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('{}/{}_{}.svg'.format(save_dir, name, idx1[0]), bbox_inches='tight')
    plt.close()


def crop(ts, perc=.1):
    seq_len = ts.shape[0]
    win_len = int(perc * seq_len)
    ts_list=[]
    for start in range(10, seq_len, win_len):
        end = start + win_len
        start = max(0, start)
        end = min(end, seq_len)
        ts_list.append(ts[start:end, ...])
    return ts_list


def crop_hier(ts, perc=.1, figsize=(10, 2), name='None'):

    seq_len = ts.shape[0]
    win_len1 = int(perc * seq_len)

    end1 = np.random.randint(0, 5)
    segments = {}

    plt.figure(figsize=figsize)
    plt.plot(x_org, color='C0', linestyle='-', lw=4)
    for i in range(4):
        start1=(end1+np.random.randint(10, 30))
        end1 = start1 + win_len1
        print(start1, end1)
        x_crop = ts[start1:end1, ...]
        segments[start1]=x_crop
        plt.plot(np.arange(start1, end1), x_crop,
             lw=8, color='C2', linestyle='-')

    plt.ylim((np.min(ts) - 0.1, np.max(ts) + 0.1))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('{}/{}.svg'.format(save_dir, name), bbox_inches='tight')
    plt.close()

    for idx in segments.keys():
        plt.figure(figsize=(1.2, 2))
        seg = segments[idx]
        plt.plot(seg, color='C2', linestyle='-', lw=8)
        plt.ylim((np.min(ts)-0.1, np.max(ts)+0.1))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('{}/{}_seg{}.svg'.format(save_dir, name, idx), bbox_inches='tight')
        plt.close()


ucr_path = './datasets'


DATAS = [
     'EpilepticSeizure',
        ]
for dataset_name in DATAS:
    np.random.seed(0)

    x_train, y_train, x_val, y_val, x_test, y_test, nb_class, _ \
                            = load_ucr2018(ucr_path, dataset_name)

    for cls in range(0, 10):
        idxs = np.where(y_train==cls)[0][:2]
        for idx in idxs:
            save_dir = './visualization/figure/crops2/{}/{}/{}'.format(dataset_name, cls, idx)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            x = x_train[idx:idx+1][:, 50:150]
            x_org = x[0]
            plot1d(x_org, figsize=(5, 2), color='C0', linestyle='-', linewidth=4,
                   save_file='{}/00_x_{}.svg'.format(save_dir, idx))
            plot1d(x_org[:50], figsize=(3, 2), color='C0', linestyle='-', linewidth=4,
                   save_file='{}/00_x_{}_P.svg'.format(save_dir, idx))
            plot1d(x_org[50:], figsize=(3, 2), color='C0', linestyle='-', linewidth=4,
                   save_file='{}/00_x_{}_F.svg'.format(save_dir, idx))


