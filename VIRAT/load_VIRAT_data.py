from __future__ import print_function
__docformat__ = 'restructedtext en'

import os
import sys
import timeit

import numpy as np
import scipy.io as io
from scipy.linalg import norm
import re
from numpy.random import randint

import errno
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import pdb

class VIRAT_Deep_Feature(data.Dataset):

    def __init__(self, root, split, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.split = split

        app_route = os.path.join(self.root,self.split[0])

        with open(app_route) as app_file:
            self.app_list = app_file.readlines()

        num_feats = len(self.app_list)
        new_idx = np.random.permutation(num_feats)
        self.train_new_idx = new_idx[:2*num_feats/3]
        self.test_new_idx = new_idx[2*num_feats/3:]

        if self.train:
            self.train_app_data = []
            self.train_labels = []
            self.count = np.zeros((6,))
            for i in range(len(self.train_new_idx)):
                app_name = self.app_list[self.train_new_idx[i]].split('  ')[0]
                name = app_name.replace('.mat','')
                label = int(self.app_list[self.train_new_idx[i]].split('  ')[1])
                label -= 1
                A = io.loadmat(app_name)['A_FEAT']
                self.train_app_data.append(A)
                self.train_labels.append(label)
                self.count[label] += 1

        else:
            self.test_app_data = []
            self.test_labels = []
            self.count = np.zeros((6,))
            for i in range(len(self.test_new_idx)):
                test_app_name = self.app_list[self.test_new_idx[i]].split('  ')[0]
                name = test_app_name.replace('.mat','')
                label = int(self.app_list[self.test_new_idx[i]].split('  ')[1])
                label -= 1
                A = io.loadmat(test_app_name)['A_FEAT']
                test_app_lengths = A.shape[1]
                self.test_app_data.append(A)
                self.test_labels.append(label)
                self.count[label] += 1


    def __getitem__(self, index):
        """
        Args:
            index (int): Indexf
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train:
            app_feature,target = self.train_app_data[index], self.train_labels[index]
            if self.transform is not None:
                app_feature = self.transform(app_feature)
            if self.target_transform is not None:
                target = self.transform(target)
            return app_feature, target
        else:
            app_feature, target = self.test_app_data[index], self.test_labels[index]
            if self.transform is not None:
                app_feature = self.transform(app_feature)

            if self.target_transform is not None:
                target = self.transform(target)
            return app_feature, target


    def __len__(self):
        if self.train:
            return len(self.train_app_data)
        else:
            return len(self.test_app_data)

    def data_count(self):
        return self.count

    def output_test_idx(self):
        return self.test_new_idx
