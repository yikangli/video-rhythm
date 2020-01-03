from __future__ import print_function
__docformat__ = 'restructedtext en'

import os
import sys

import numpy as np
import scipy.io as io

import errno
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import pdb
#from .utils import download_url, check_integrity

class BF_Deep_Feature(data.Dataset):

    def __init__(self, root, train_split, test_split, train=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.train_split = train_split
        self.test_split = test_split

        if self.train:
            app_route = os.path.join(self.root,self.train_split[0])
            label_route = os.path.join(self.root,self.train_split[1])

            with open(app_route) as app_file:
                #self.train_app_list = app_file.readlines()
                self.app_list = app_file.readlines()

            with open(label_route) as label_file:
                self.label_list = label_file.readlines()
            
        else:
            app_route = os.path.join(self.root,self.test_split[0])
            label_route = os.path.join(self.root,self.test_split[1])

            with open(app_route) as app_file:
                self.app_list = app_file.readlines()

            with open(label_route) as label_file:
                self.label_list = label_file.readlines()

        #self.train_app_data = []
        #self.train_labels = []
        self.labels = []
        #self.count = np.zeros((6,))
        for i in range(len(self.label_list)):
            #app_name = self.app_list[i].split()[0]
            label_name = self.label_list[i].split()[0]
            #A = io.loadmat(app_name)['rgb_matrix']
            C = io.loadmat(label_name)['label_matrix']
            C = int(C[0,1]) - 1
            #self.train_app_data.append(A)
            #self.train_labels.append(C)
            self.labels.append(C)
        #self.train_labels = np.array(self.train_labels, dtype=int)
        self.labels = np.array(self.labels, dtype=int)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        app_feature = self.load_frames(self.app_list[index])
        target = np.array(self.labels[index])
        app_length = [app_feature.shape[1]]
        return app_feature, app_length, target
        #if self.train:
        #    app_feature = self.load_frames(self.train_app_list[index])
        #    target = np.array(self.train_labels[index])
        #    app_length = [app_feature.shape[1]]
        #   return app_feature, app_length, target
        #else:
        #    app_feature = self.load_frames(self.test_app_list[index])
        #    target = np.array(self.test_labels[index])
        #    app_length = [app_feature.shape[1]]
        #    return app_feature, app_length, target

    def load_frames(self, file_dir):
        """
        Loading frame features
        """
        #pdb.set_trace()
        #frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        app_name = file_dir.split()[0]
        frames = io.loadmat(app_name)['rgb_matrix']
        frames = frames[10:-10]
        #frames = frames[::2]
        frames = np.transpose(frames)
        return frames

    def __len__(self):
        return len(self.app_list)

'''
if __name__ == "__main__":
    import BF_dataloader
    root = './'
    train_split = ['App_train_TSN_VGGlist.txt','Label_train_TSN_list.txt']
    test_split = ['App_test_TSN_VGGlist.txt','Label_test_TSN_list.txt']
    train_flag = True
    train_data = BF_Deep_Feature(root, train_split, test_split, train=train_flag)
    pdb.set_trace()
    train_loader = BF_dataloader.DataLoader(train_data, batch_size=1, shuffle=True)
    for i, (app_inputs,app_lengths,targets) in enumerate(train_loader):
        inputs = app_inputs
        labels = targets
        print(inputs.shape())
        print(labels)

        if i == 1:
            break
'''