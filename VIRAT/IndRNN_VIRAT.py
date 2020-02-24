from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.rnn as rnn_utils

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy
import numpy.matlib

#from models import *
from utils import progress_bar
import timeit
from tqdm import tqdm
from torch.autograd import Variable

import load_VIRAT_data
import VIRAT_Dataloader
import indrnn

import pdb

use_cuda = torch.cuda.is_available()
num_input = 4096
num_hidden1 = 250
mlp_hidden1 = 50
mlp_hidden2 = 1
num_mlp = 1000
num_emb = 100
class_num = 6
batch_size = 1
margin = 0.1
step_size = 2

start_epoch = 1  # start from epoch 0 or last checkpoint epoch

root = './'
split = ['App_list.txt']

# Data
print('==> Preparing data..')

trainset = load_VIRAT_data.VIRAT_Deep_Feature(root, split, train=True)
trainloader = VIRAT_Dataloader.DataLoader(trainset, batch_size=batch_size, shuffle=True)
print('training sample is %d', len(trainset))

testset = load_VIRAT_data.VIRAT_Deep_Feature(root, split, train=False)
testloader = VIRAT_Dataloader.DataLoader(testset, batch_size=batch_size, shuffle=False)

events = ('CAV', 'GIV', 'GOV', 'LAV', 'OAV', 'UAV')

def l2norm(X):
    """
    Compute L2 norm, row-wise
    """
    norm = torch.sqrt(torch.pow(X, 2).sum(1))
    X /= norm[:, None]

    return X


# Model
class VENet(nn.Module):
    def __init__(self):
        super(VENet, self).__init__()
        self.IndRNN = indrnn.IndRNN(input_size=num_input,hidden_size=num_hidden1, n_layer=2, batch_first=False, nonlinearity="relu",bidirectional=True)
        self.mlp_pooling1 = nn.Linear(2*num_hidden1,mlp_hidden1)
        self.mlp_pooling2 = nn.Linear(mlp_hidden1,mlp_hidden2)

        self.lstm_app = nn.GRU(input_size=num_input,hidden_size=num_input/2, num_layers=1, bias=True)
        self.fc1 = nn.Linear(num_input/2, num_emb)
        self.fc2 = nn.Linear(num_emb,class_num)
        nn.init.xavier_normal_(self.fc1.weight, gain=numpy.sqrt(2))
        nn.init.xavier_normal_(self.fc2.weight, gain=numpy.sqrt(2))

    def forward(self, app_x, app_lengths, hidden_app=None):
        #pdb.set_trace()
        app_x = F.relu(app_x)
        X_pooled,hidden_pooled = self.IndRNN(app_x,hidden_app)  
        X_pooled = F.relu(self.mlp_pooling1(X_pooled.squeeze()))
        Pooled_output = torch.sigmoid(self.mlp_pooling2(X_pooled))

        A = Variable(torch.ones(Pooled_output.size(0)).cuda())
        B = Variable(torch.zeros(Pooled_output.size(0)).cuda())
        Pooled_list = torch.where(Pooled_output[:,0]>0.5,A,B)
        X_pooled = (Pooled_list.squeeze()).nonzero()
        if not X_pooled.size(0):
            x_input = app_x[0,:,:]
            X_pooled = Variable(torch.zeros(1).cuda())
            packed_input = rnn_utils.pack_padded_sequence(x_input.unsqueeze(1),[len(X_pooled)])
        else:
            x_input = app_x[X_pooled[:,0],:,:]
            packed_input = rnn_utils.pack_padded_sequence(x_input,[len(X_pooled)])

        X_app,hidden_app = self.lstm_app(packed_input,hidden_app)
        X_app,_ = rnn_utils.pad_packed_sequence(X_app)

        #pdb.set_trace()
        X_app = X_app[-1,:,:]
        X_emb = F.relu(self.fc1(X_app))
        norm = torch.norm(X_emb, p=2, dim=1, keepdim=True)
        X_emb = X_emb.div(norm)
        outputs = self.fc2(X_emb)

        return outputs, Pooled_output, X_pooled


def loss_regu(Pooled_output,X_pooled):
    regu1 = torch.abs(torch.mean(Pooled_output) - 0.25)
    penalty = float(X_pooled.size(0))/Pooled_output.size(0)
    return regu1, penalty



for k in range(1):
    print('\nCross Fold %d' % k)
    best_acc = 0
    net = VENet()

    if use_cuda:
        net.cuda()
        cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=20, verbose=True)
    
    # Training
    for epoch in range(start_epoch, start_epoch+750):
        # train(epoch)
        print('\nEpoch: %d' % epoch)
        net.train()
        en_loss = 0
        Regular1 = 0
        Regular2 = 0
        Pen = 0
        total = 0
        train_emb_list = []
        for batch_idx, (app_inputs, app_lengths, targets) in enumerate(trainloader):
            app_inputs, targets = torch.from_numpy(app_inputs), torch.from_numpy(targets)
            app_inputs, targets = app_inputs.type(torch.FloatTensor), targets.type(torch.LongTensor)
            if use_cuda:
                app_inputs, targets = app_inputs.cuda(),  targets.cuda()
            optimizer.zero_grad()
            app_inputs, targets = Variable(app_inputs), Variable(targets)
            outputs, Pooled_output, X_pooled= net(app_inputs, app_lengths)

            entropy_loss = criterion(outputs, targets.type(torch.cuda.LongTensor))
            regu1,penalty = loss_regu(Pooled_output,X_pooled)
            loss = entropy_loss + 2*regu1 

            loss.backward()
            optimizer.step()
            # scheduler.step()

            en_loss += entropy_loss.item()
            Regular1 += regu1.item()
            Pen += penalty
            progress_bar(batch_idx, len(trainloader), 'Cross Entropy Loss: %.3f, Regu1: %.3f, Sparsity: %.3f' % (en_loss/float(batch_idx+1),Regular1/float(batch_idx+1),Pen/float(batch_idx+1)))
            
        scheduler.step((loss)/(batch_idx+1))
        if numpy.mod(epoch, 1) == 0:
            test_binary_1 = []
            test_binary_2 = []
            test_binary_3 = []
            test_binary = []
            test_emb_list = []
            y_test = []
            totime = 0
            totime1 = 0
            totime2 = 0
            totime3 = 0
            S_time = timeit.default_timer()
            for test_app_inputs, test_app_lengths, test_targets in tqdm(testloader):
                test_app_inputs, test_targets = torch.from_numpy(test_app_inputs), torch.from_numpy(test_targets)
                test_app_inputs, test_targets = test_app_inputs.type(torch.FloatTensor), test_targets.type(torch.LongTensor)
                A_app = test_app_inputs[:test_app_lengths[0]/3,:,:]
                B_app = test_app_inputs[test_app_lengths[0]/3:2*test_app_lengths[0]/3,:,:]
                C_app = test_app_inputs[2*test_app_lengths[0]/3:,:,:]

                A1_app = A_app[::2,:,:]
                B1_app = B_app
                C1_app = C_app[::5,:,:]

                test_app_inputs_1 = torch.cat((A1_app, B1_app, C1_app),dim=0)
                test_app_lengths1 = [test_app_inputs_1.shape[0]]

                A2_app = A_app[::5,:,:]
                B2_app = B_app[::2,:,:]
                C2_app = C_app
                test_app_inputs_2 = torch.cat((A2_app, B2_app, C2_app),dim=0)
                test_app_lengths2 = [test_app_inputs_2.shape[0]]

                random_pick1_app = torch.randperm(test_app_lengths[0]/3)
                random_pick1_app = random_pick1_app[:test_app_lengths[0]/(6)]
                random_pick1_app,_ = torch.sort(random_pick1_app)
                random_pick2_app = torch.randperm(test_app_lengths[0]/3)
                random_pick2_app = random_pick2_app[:test_app_lengths[0]/(6)]
                random_pick2_app,_ = torch.sort(random_pick2_app)
                random_pick3_app = torch.randperm(test_app_lengths[0]/3)
                random_pick3_app = random_pick3_app[:test_app_lengths[0]/(6)]
                random_pick3_app,_ = torch.sort(random_pick3_app)

                A3_app = A_app[random_pick1_app,:,:]
                B3_app = B_app[random_pick2_app,:,:]
                C3_app = C_app[random_pick3_app,:,:]
                test_app_inputs_3 = torch.cat((A3_app,B3_app,C3_app),dim=0)
                test_app_lengths3 = [test_app_inputs_3.shape[0]]

                test_app_inputs= test_app_inputs.type(torch.FloatTensor)
                test_app_inputs_1 = test_app_inputs_1.type(torch.FloatTensor)
                test_app_inputs_2 = test_app_inputs_2.type(torch.FloatTensor)
                test_app_inputs_3 = test_app_inputs_3.type(torch.FloatTensor)

                if use_cuda:
                    test_app_inputs = test_app_inputs.cuda()
                    test_app_inputs_1 = test_app_inputs_1.cuda()
                    test_app_inputs_2 = test_app_inputs_2.cuda()
                    test_app_inputs_3 = test_app_inputs_3.cuda()
                    
                test_app_inputs = Variable(test_app_inputs)
                test_app_inputs_1 = Variable(test_app_inputs_1)
                test_app_inputs_2 = Variable(test_app_inputs_2)
                test_app_inputs_3 = Variable(test_app_inputs_3)
                
                #pdb.set_trace()
                start_time = timeit.default_timer()
                test_outputs,_,_ = net(test_app_inputs, test_app_lengths)
                stop_time = timeit.default_timer()
                totime += stop_time - start_time

                start_time = timeit.default_timer()
                test_outputs_1,_ ,_= net(test_app_inputs_1, test_app_lengths1)
                stop_time = timeit.default_timer()
                totime1 += stop_time - start_time

                start_time = timeit.default_timer()
                test_outputs_2,_,_ = net(test_app_inputs_2, test_app_lengths2)
                stop_time = timeit.default_timer()
                totime2 += stop_time - start_time

                start_time = timeit.default_timer()
                test_outputs_3,_ ,_= net(test_app_inputs_3, test_app_lengths3)
                stop_time = timeit.default_timer()
                totime3 += stop_time - start_time

                test_outputs = test_outputs.data.cpu().numpy()
                test_outputs_1 = test_outputs_1.data.cpu().numpy()
                test_outputs_2 = test_outputs_2.data.cpu().numpy()
                test_outputs_3 = test_outputs_3.data.cpu().numpy()

                test_binary.append(test_outputs)
                test_binary_1.append(test_outputs_1)
                test_binary_2.append(test_outputs_2)
                test_binary_3.append(test_outputs_3)

                y_test.append(test_targets)

            test_binary = numpy.asarray(test_binary)
            test_binary_1 = numpy.asarray(test_binary_1)
            test_binary_2 = numpy.asarray(test_binary_2)
            test_binary_3 = numpy.asarray(test_binary_3)

            test_binary = numpy.reshape(test_binary, (test_binary.shape[0]*test_binary.shape[1], test_binary.shape[2]))
            test_binary_1 = numpy.reshape(test_binary_1, (test_binary_1.shape[0]*test_binary_1.shape[1], test_binary_1.shape[2]))
            test_binary_2 = numpy.reshape(test_binary_2, (test_binary_2.shape[0]*test_binary_2.shape[1], test_binary_2.shape[2]))
            test_binary_3 = numpy.reshape(test_binary_3, (test_binary_3.shape[0]*test_binary_3.shape[1], test_binary_3.shape[2]))

            y_test = numpy.asarray(y_test)

            predict_label = numpy.argmax(test_binary, axis=1)
            predict_label_1 = numpy.argmax(test_binary_1, axis=1)
            predict_label_2 = numpy.argmax(test_binary_2, axis=1)
            predict_label_3 = numpy.argmax(test_binary_3, axis=1)

            acc = len(numpy.where((predict_label-y_test) == 0)[0]) / float(len(y_test))
            acc_1 = len(numpy.where((predict_label_1-y_test) == 0)[0]) / float(len(y_test))
            acc_2 = len(numpy.where((predict_label_2-y_test) == 0)[0]) / float(len(y_test))
            acc_3 = len(numpy.where((predict_label_3-y_test) == 0)[0]) / float(len(y_test))
            
            print('epoch %d and Accuracy 0 is %.6f; Accuracy 1 is %.6f; Accuracy 2 is %.6f; Accuracy 3 is %.6f\n' % (epoch, acc, acc_1,acc_2,acc_3))
            print("Time: %.6f, Time1: %.6f, Time2: %.6f, Time3: %.6f \n" %(totime, totime1, totime2, totime3))
            SS_time = timeit.default_timer()
            print('Execution Time is '+ str(SS_time - S_time) + "\n")

            if acc > best_acc:
                print('Saving..')
                state = {
                    'net': net,
                    'ACC1': acc,
                    'state_dict': net.state_dict(),
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt_IndRNN_VIRAT.t7')
                best_acc = acc

            if epoch == 750:
                print('Saving..')
                state = {
                    'net': net,
                    'ACC1': acc,
                    'state_dict': net.state_dict(),
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt_IndRNN_VIRAT_last.t7')
                best_acc = acc

# test(epoch)
