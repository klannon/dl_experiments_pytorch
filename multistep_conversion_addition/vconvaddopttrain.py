#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt
import collections
from collections import OrderedDict
import argparse

## User Inputs

parser = argparse.ArgumentParser(description='Train ANN auto-encoder.')
parser.add_argument('trainfile',
                        help='File name for training set')

parser.add_argument('-o','--out', dest='outbase', metavar='OUTBASE',
                        help='File name base (no extension)')

parser.add_argument('-b','--batch-size',
                        default=250, type=int, help='Batch size for data during training and testing')

parser.add_argument('-n','--num-epochs', default=200, type=int, 
                        help='Number of epochs in a training cycle')

parser.add_argument('-v','--valid-size', type=int, 
                        help='Size of the validation set')

args = parser.parse_args()

## User Inputs

infile = args.trainfile
seed = np.random.randint(low=0, high=1000000, size=1)
H = 270
trials = 3
batch_size = args.batch_size
valid_size = args.valid_size
n_epochs = args.num_epochs

use_cuda = torch.cuda.is_available()
print(use_cuda)

## Loading Data

npfile = np.load(infile)
inputs = npfile['inputs']
outputs = npfile['outputs']

npfile.close()

# standardizing inputs, outputs and coverting to tensors
inputMeans = inputs.mean(axis=0)
inputStdDevs = inputs.std(axis=0)
inputs_np = (inputs-inputMeans)/inputStdDevs
inputs_tens = torch.from_numpy(inputs_np).float()
inputs_np = None
inputs = inputs_tens.cuda()
inputs_tens = None

outputMeans = outputs.mean(axis=0)
outputStdDevs = outputs.std(axis=0)
outputs_np = (outputs-outputMeans)/outputStdDevs
outputs_tens = torch.from_numpy(outputs_np).float()
outputs_np = None
outputs = outputs_tens.cuda()
outputs_tens = None

# generating a TensorDataset for training
train_set = torch.utils.data.TensorDataset(inputs, outputs)

# creating the model

# defining model class and forward process

class SharedModel(torch.nn.Module):
    def __init__(self,H):
        super(SharedModel,self).__init__()
        self.linear1 = torch.nn.Linear(3,H)
        torch.nn.init.kaiming_normal_(self.linear1.weight,a=0.25, nonlinearity='leaky_relu')
        torch.nn.init.constant_(self.linear1.bias,0)
        self.activation = torch.nn.PReLU()
        self.linear2 = torch.nn.Linear(H,3)
        torch.nn.init.kaiming_normal_(self.linear2.weight,a=0.25, nonlinearity='leaky_relu')
        torch.nn.init.constant_(self.linear2.bias,0)
        self.linear3 = torch.nn.Linear(6,3)
        torch.nn.init.kaiming_normal_(self.linear3.weight,a=0.25, nonlinearity='leaky_relu')
        torch.nn.init.constant_(self.linear3.bias,0)
        self.linear4 = torch.nn.Linear(3,3)
        torch.nn.init.kaiming_normal_(self.linear4.weight,a=0.25, nonlinearity='leaky_relu')
        torch.nn.init.constant_(self.linear4.bias,0)

    def forward(self,x):
        vars = torch.chunk(x,2,1)
        conv_results = []
        for v in vars:
            o = self.linear1(v)
            o = self.activation(o)
            o = self.linear2(o)
            conv_results.append(o)
        x_cat = torch.cat(conv_results,1)
        o = self.linear3(x_cat)
        o = self.activation(o)
        x_pred = self.linear4(o)
        return x_pred     

model = SharedModel(H)
model = model.cuda()

## Function Definitons

# defines the training process for one epoch, returns training loss for given epoch
def train(model, loader, optimizer, criterion):
    running_train_loss = 0.0

    # put model into train mode
    model.train()

    for batch_idx, (inputs, outputs) in enumerate(loader):
        inputs_var = inputs
        outputs_var = outputs
        
        # get model output & loss for each given input
        model_outputs = model(inputs_var)
        loss = criterion(model_outputs, outputs_var)

        # record cummulative loss
        running_train_loss += loss.item()

        # gradient, optimizer steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_train_loss


# defines the validation process for one epoch, returns validation loss for given epoch
def validate(model, loader, criterion):
    running_valid_loss = 0.0

    # put model in evaluation mode
    model.eval()

    for batch_idx, (inputs, outputs) in enumerate(loader):
        with torch.no_grad():
            inputs_var = inputs
            outputs_var = outputs

            # get model output & loss for each given input
            model_outputs = model(inputs_var)
            loss = criterion(model_outputs, outputs_var)

        # record cummulative loss
        running_valid_loss += loss.item()

    return running_valid_loss


# runs training and validation process over all epochs, returns results
def run_training(model, modelpath, trainset, validsize, numepochs, batchsize, seed):
    # set seed
    torch.manual_seed(seed)

    # create validation split
    indices = torch.randperm(len(trainset))
    train_indices = indices[:len(indices) - valid_size]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_indices = indices[len(indices) - valid_size:]
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    # define data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler)
  
    # set criterion, optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # store results
    best_model = copy.deepcopy(model.state_dict())
    train_loss_results = []
    valid_loss_results = []
    epochs = []
    
    # train model
    for epoch in enumerate(range(n_epochs)):
        trainloss = train(model=model,loader=train_loader,criterion=criterion,optimizer=optimizer)
        print('train loss for epoch {index} attained: {loss}'.format(index=epoch[0], loss=trainloss))
        
        validloss = validate(model=model,loader=valid_loader,criterion=criterion)
        print('valid loss for epoch {index} attained: {loss}'.format(index=epoch[0], loss=validloss))
        
        train_loss_results.append(trainloss)
        valid_loss_results.append(validloss)
        epochs.append(epoch[0]+1)
        
        # check if model is the best, save if best
        if epoch[0] == 0:
            bestloss = validloss

        if epoch[0] > 0:
            if validloss < bestloss:
                bestloss = validloss
                best_model = copy.deepcopy(model)
                best_epoch = epoch[0]
                print('new best model saved')
                
        print('epoch {index} done'.format(index=epoch[0]))
        
    print('finished looping epochs')
    print('best valid loss = {}, epoch {}'.format(bestloss, best_epoch))

    # load and save the best model
    torch.save(best_model, modelpath)
    print('best model loaded and saved')
    
    return

## Run Training Function

for trial in range(trials):
    # train the model
    run_training(model=model, modelpath='vconvaddopt_270_270_6_3nodes_{}'.format(trial), trainset=train_set, validsize=valid_size, 
                numepochs=n_epochs, batchsize=batch_size, seed=seed)

    print('Finished!')