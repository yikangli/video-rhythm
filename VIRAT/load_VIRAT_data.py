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
#from .utils import download_url, check_integrity
import pdb

class VIRAT_Deep_Feature(data.Dataset):

    def __init__(self, root, split, train=True,transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.split = split

        if self.train:
            app_route = os.path.join(self.root,self.split[0])
            with open(app_route) as app_file:
                self.app_list = app_file.readlines()

        else:
            app_route = os.path.join(self.root,self.split[1])
            with open(app_route) as app_file:
                self.app_list = app_file.readlines()

    def __getitem__(self, index):
        """
        Args:
            index (int): Indexf
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        app_feature = self.load_frames(self.app_list[index])
        target = int(self.app_list[index].split('  ')[1])
        target -= 1
        target = np.array(target)
        app_length = [app_feature.shape[1]]
        return app_feature, app_length, target

    def load_frames(self, file_dir):
        """
        Loading frame features
        """
        app_name = file_dir.split('  ')[0]
        frames = io.loadmat(app_name)['A_FEAT']
        return frames

    def __len__(self):
        return len(self.app_list)
