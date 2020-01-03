from __future__ import print_function

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.rnn as rnn_utils

import torchvision
import torchvision.transforms as transforms

import os
import scipy.io as sio
import argparse
import numpy as np
import numpy.matlib

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchvision import models

import pdb

app_route = '/home/local/ASUAD/yikangli/Documents/sub_activity_dataset/BreakFast/Frame/'
save_app_root = '/home/local/ASUAD/yikangli/Documents/sub_activity_dataset/BreakFast/cache_deep/deep_frame/'

mean_app = np.asarray((0.4914, 0.4822, 0.4465))
std_app = np.asarray((0.2023, 0.1994, 0.2010))

transform_app = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])

video_per = sorted(os.listdir(app_route))

use_cuda = torch.cuda.is_available()

# Model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
	#self.model = models.vgg16_bn(pretrained=True)
	#new_classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])
        #self.model.classifier = new_classifier
        #pdb.set_trace()
	model = models.resnet152(pretrained=True)
        modules=list(model.children())[:-1]
        self.app_conv=nn.Sequential(*modules)

    def forward(self, app_x):
        #pdb.set_trace()
        app_x = app_x.view(1, app_x.shape[2], app_x.shape[1], app_x.shape[0])
        #pdb.set_trace()
        #app_feats = self.model(app_x)
        app_feats = self.app_conv(app_x)
        return app_feats.squeeze()

net = Net()

if use_cuda:
    net.cuda()
    cudnn.benchmark = True

#pdb.set_trace()
for p in video_per:
    app_person_dir = os.path.join(app_route,p+'/')
    app_cam_num = sorted(os.listdir(app_person_dir))
    save_name_app_per = os.path.join(save_app_root,p+'/')
    for q in app_cam_num:
        print('Compute the %s camera in %s sample in dataset' %(q,p))
        app_cam_dir = os.path.join(app_person_dir,q+'/')
        video_name = sorted(os.listdir(app_cam_dir))
        save_name_app_cam = os.path.join(save_name_app_per,q+'/')
        if not os.path.exists(save_name_app_cam):
            os.makedirs(save_name_app_cam)
        for v in video_name:
            app_video_clips = os.path.join(app_cam_dir,v)
            app_feats = sio.loadmat(app_video_clips)['rgb_matrix']
            app_feats = ((app_feats/255.)-mean_app)/std_app
            app_feats = torch.from_numpy(app_feats)
            app_feats = app_feats.type(torch.FloatTensor)
            app_feats = Variable(app_feats.cuda())
            APP = np.zeros((app_feats.shape[0],2048))
            for i in range(app_feats.shape[0]):
                app_output = net(app_feats[i,:,:,:])
                APP[i,:] = app_output.data.cpu().numpy()
            save_name_app_video = os.path.join(save_name_app_cam,v)
            sio.savemat(save_name_app_video, {'rgb_matrix': APP})

