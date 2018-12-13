import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets as dset

def loadCifar10(dataroot, download=True, 
				image_size=32,
				train_batch_size=64, 
				valid_batch_size=128, 
				test_batch_size=128, 
				transform=True,
				workers=2,
				valid_size=1000,
				):
	'''Load cifar10 data
	We will download the data or you can assign the dataroot.

	Return:
		return with train_dataloader,valid_dataloader,test_dataloader,
	'''

	# train data loading and preprocessing
	if transform:
		train_transform = transforms.Compose([
		                           transforms.RandomCrop(image_size, padding=4),
		                           transforms.RandomHorizontalFlip(),
		                           transforms.ToTensor(),
		                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		                       ])
		test_transform = transforms.Compose([
		                           transforms.ToTensor(),
		                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		                       ])
	else:
		train_transform = None
		test_transform = None

	cifar10data = dset.CIFAR10(root=dataroot, train=True, download=download,
		                       transform=train_transform)
	# Test data and validation data
	# test data loading and preprocessing
	test_dataset = dset.CIFAR10(root=dataroot, train=False, download=False,
		                       transform=test_transform)

	# splite data into three parts
	valid_dataset = []
	for i in range(valid_size):
	    valid_dataset.append(test_dataset[i])

	train_dataloader = torch.utils.data.DataLoader(cifar10data,
	                                        batch_size=train_batch_size,
	                                        shuffle=True, num_workers=workers)
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
	                                        batch_size=valid_batch_size,
	                                        shuffle=False, num_workers=workers)
	# create test dataloader
	test_dataloader = torch.utils.data.DataLoader(test_dataset,
	                                        batch_size=test_batch_size,
	                                        shuffle=False, num_workers=workers)
	return train_dataloader, valid_dataloader, test_dataloader

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def getParametersNumber(net):
	params = list(net.parameters())
	params_number = 0
	parameterslist = []
	for i in params:
	    layer_params = 1
	    for j in i.size():
	        layer_params *= j        
	        params_number += layer_params
	    #print("#Parameters in Module{}:{}".format(i.size(), layer_params))
	    parameterslist.append((i.size(),layer_params))
	print("Total Parameters:%d" % (params_number))