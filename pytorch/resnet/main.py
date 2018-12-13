from __future__ import print_function
import argparse
import random
import numpy as np
import time
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets as dset

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import utils
from model import ResNet18
from utils import loadCifar10, weight_init, getParametersNumber

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, required=False, help='cifar10')
parser.add_argument('-dataroot', type=str, required=False, help='path to dataset')
parser.add_argument('-download', type=bool, required=False, help='path to dataset')
parser.add_argument('-workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('-trainBatchSize', type=int, default=64, help='input train batch size')
parser.add_argument('-validBatchSize', type=int, default=128, help='input validation batch size')
parser.add_argument('-testBatchSize', type=int, default=128, help='input test batch size')
parser.add_argument('-imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('-nc', type=int, default=3, help='input image channels')
parser.add_argument('-ncf', type=int, default=64)
parser.add_argument('-epochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('-ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('-adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-lrdecaytype', type=int, default=1, help='learning rate decay:1 for each epoch variation, 2 for epoch range variation')
parser.add_argument('-decay', type=float, default=0.98, help='lr decay for each epoch')
parser.add_argument('-beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('-baseline', type=float, default=0.85, help='baseline for saving trained model')
parser.add_argument('-train', type=bool, default=True, help='Train or test')
parser.add_argument('-outpdir', type=str, default='./ckpt', help='where to save your model')
parser = parser.parse_args()


device = torch.device('cuda' if (torch.cuda.is_available() and parser.ngpu > 0) else 'cpu')
print('Training use:',device)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
train_dataloader, valid_dataloader, test_dataloader =\
	loadCifar10(dataroot=parser.dataroot, 
				download=False, 
                image_size=parser.imageSize,
				train_batch_size=parser.trainBatchSize, 
				valid_batch_size=parser.validBatchSize, 
				test_batch_size=parser.testBatchSize, 
				transform=True)

if not parser.train:
	net = torch.load('trainedModel.pt')
	net = net.to(device)
	#net = ResNet18().to(device)

	if (device.type == 'cuda') and ngpu > 1:
	    net = nn.DataParallel(net, list(range(ngpu)))
	# test set
	with torch.no_grad():
	    correct = 0.0
	    total = 0.0
	    test_loss = 0.0
	    for i, data in enumerate(test_dataloader):
	        net.eval()
	        input,label = data[0].to(device), data[1].to(device)
	        output = net(input)
	        loss = criterion(output, label) 
	        test_loss += loss.item()  # scalar
	        # get results
	        predicted = torch.argmax(output.data, 1)
	        total += label.size(0)
	        correct += (predicted == label).sum().item()
	        
	    test_acc = 1.0 * correct / total
	    print('TestLoss：%f\tTestAcc over %d images:%.4f' % (test_loss, total, test_acc))

#################################################
# train your model
#################################################

# save train log
train_losses = []
train_accs = []
val_losses = []
val_accs = []
test_accs = []

lr = parser.lr
lrdecaytype = parser.lrdecaytype
decay = parser.decay

valid_baseline = parser.baseline
test_baseline = parser.baseline
num_epochs = parser.epochs
ngpu = parser.ngpu



iters = 0
# write into file
log_dir = parser.outpdir + "/resnetlogs"
ckpt_dir = parser.outpdir

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# Load model
net = ResNet18().to(device)
if (device.type == 'cuda') and ngpu > 1:
    net = nn.DataParallel(net, list(range(ngpu)))
#print(net)

# set up your loss function and optmizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=parser.lr)#, momentum=momentum)


print('#######################Start Training#######################')
with open('resnet_timelogs.csv', 'w') as timefile:
    with open('resnet_acc.csv', 'w') as accfile:
        ##############################################
        ##    Train at every epoch                  ##
        ##############################################
        for epoch in range(num_epochs):
            # train with all training data
            print('Epoch:%d' %(epoch))
            start_epoch_time = time.time()
            train_loss = 0.0
            train_accuracy = 0.0
            net.train()
            correct = 0.0
            total = 0.0
            # set learning rate wrt. epoch
            if lrdecaytype == 1:
            	lr = lr * decay
            	optimizer = optim.Adam(net.parameters(), lr=lr)#, momentum=momentum)
            elif lrdecay == 2 and epoch % 60 == 0:
                lr = lr * 0.1
                optimizer = optim.Adam(net.parameters(), lr=lr)#, momentum=momentum)
            ############################################
            ## train in each batch of training data
            for i,data in enumerate(train_dataloader, 0):
                input,label = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                
                # Forward pass 
                output = net(input)
                # calculate accuracy
                loss = criterion(output, label) # criterion(input=(N,C),target=(N,))
                # backward and update
                loss.backward()
                optimizer.step()

                
                # calculate accuracy
                y_pred = torch.argmax(input=output, dim=1)  # predicted label
                y_pred_right = (y_pred == label).sum().item() # scalar
                # calculate epoch accuracy
                total += y_pred.size(0)  # scalar
                correct += y_pred_right
                
                train_accuracy = correct / total
                train_loss += loss.item()
                
                # Print every 100 iters
                if i % 100 == 0:
                    print('[Epoch%d/%d][Batchiter%d/%d]\titer:%d\tTrainEpochAcc: %.4f\tTrainLoss: %f'\
                         % (epoch, num_epochs, i, len(train_dataloader), iters, train_accuracy, train_loss/(i+1)))
                iters += 1
            train_accs.append(train_accuracy)
            
            # save logfile temp
            end_epoch_time = time.time()
            epoch_take_time = end_epoch_time - start_epoch_time
            timefile.write('%d,%f\n' % (epoch, epoch_take_time))
            timefile.flush()
            
            print('##################################################')
            print('----This epoch takes time: %dmin%.2fs' % (epoch_take_time / 60,epoch_take_time % 60))
            print('##################################################')
            # validation
            print("Validation...")
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                validation_loss = 0.0
                for i, data in enumerate(valid_dataloader):
                    net.eval()
                    input,label = data[0].to(device), data[1].to(device)
                    output = net(input)
                    loss = criterion(output, label) 
                    validation_loss += loss.item()  # scalar
                    # get results
                    predicted = torch.argmax(output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    
                validation_acc = 1.0 * correct / total
                print('[%d/%d]ValBatchLoss: %.6f\tValAcc over %d images:%.4f' % (epoch, num_epochs, validation_loss/(i+1), total, validation_acc))
                # update logs
                val_losses.append(validation_loss/len(valid_dataloader))
                val_accs.append(validation_acc)
            
            # test set
            print('Testing...')
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                test_loss = 0.0
                for i, data in enumerate(test_dataloader):
                    net.eval()
                    input,label = data[0].to(device), data[1].to(device)
                    output = net(input)
                    loss = criterion(output, label) 
                    test_loss += loss.item()  # scalar
                    # get results
                    predicted = torch.argmax(output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                test_acc = 1.0 * correct / total
                print('TestBatchLoss：%f\tTestAcc over %d images:%.4f' % (test_loss, total, test_acc))
                test_accs.append(test_acc)

            accfile.write('%f,%f,%f\n' % (train_accuracy, validation_acc, test_acc))
            accfile.flush()
            
            # save model
            if (validation_acc > valid_baseline) or (test_baseline < test_acc) or (iters % 200 == 0) or ((epoch == num_epochs-1) and (i == len(train_dataloader)-1)):
                valid_baseline = validation_acc
                test_baseline = test_acc
                with torch.no_grad():
                    torch.save(net.state_dict(), ckpt_dir + '/state_dict_resnet_epoch=%d_val_acc=%.4f_test_acc=%.4f.pt' % (epoch, validation_acc,test_acc))
                    torch.save(net, ckpt_dir + '/resnet_epoch=%d_val_acc=%.4f_test_acc=%.4f.pt' % (epoch, validation_acc,test_acc))
            

