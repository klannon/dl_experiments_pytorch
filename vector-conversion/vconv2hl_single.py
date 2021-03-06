#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import copy
import collections
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(description='Train ANN auto-encoder.')
parser.add_argument('trainfile',
                        help='File name for training set')

parser.add_argument('testfile',
                        help='File name for test set')

parser.add_argument('-o','--out', dest='outbase', metavar='OUTBASE',
                        help='File name base (no extension)')

parser.add_argument('-l1','--layer1-nodes',
                        type=int, help='Number of nodes in the first hidden layer')
    
parser.add_argument('-l2','--layer2-nodes',
                        type=int, help='Number of nodes in the second hidden layer')

parser.add_argument('-t','--trials',
                        default=5, type=int, help='Number of trials per model')

parser.add_argument('-b','--batch-size',
                        default=250, type=int, help='Batch size for data during training and testing')

parser.add_argument('-n','--num-epochs', default=200, type=int, 
                        help='Number of epochs in a training cycle')

parser.add_argument('-v','--valid-size', type=int, 
                        help='Size of the validation set')

args = parser.parse_args()

## User Inputs

infile = args.trainfile
infile_test = args.testfile
seed = np.random.randint(low=0, high=1000000, size=1)

trials = args.trials
batch_size = args.batch_size
valid_size = args.valid_size
n_epochs = args.num_epochs
hl1_nodes = args.layer1_nodes
hl2_nodes = args.layer2_nodes

outfile = args.outbase

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

## Function Definitions

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
            
        validloss = validate(model=model,loader=valid_loader,criterion=criterion)
            
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
                    
    # load and save the best model
    torch.save(best_model, modelpath)
        
    return


network = OrderedDict([])

layer_1 = nn.Linear(int(3), int(hl1_nodes))
torch.nn.init.kaiming_uniform_(layer_1.weight, a=0.25, nonlinearity='leaky_relu')
torch.nn.init.constant_(layer_1.bias,0)

network = OrderedDict([('lin1', layer_1),('relu1', nn.PReLU())]) 

layer_2 = nn.Linear(int(hl1_nodes), int(hl2_nodes))  
torch.nn.init.kaiming_uniform_(layer_2.weight, a=0.25, nonlinearity='leaky_relu')
torch.nn.init.constant_(layer_2.bias,0)

network['lin2'] = layer_2

layer_3 = nn.Linear(int(hl2_nodes), int(3))  
torch.nn.init.kaiming_uniform_(layer_3.weight, a=0.25, nonlinearity='leaky_relu')
torch.nn.init.constant_(layer_3.bias,0)

network['lin3'] = layer_3  

model = nn.Sequential(network)

model = model.cuda()

for trial in range(trials):
    run_training(model = model, modelpath = '{}_{}'.format(outfile,trial), trainset = train_set, validsize = valid_size, numepochs = n_epochs, batchsize = batch_size, seed = seed)
    print('Training complete: {}, {} node model, trial {}'.format(hl1_nodes,hl2_nodes,trial))


## Testing the models

npfile_test = np.load(infile_test)
inputs_test = npfile_test['inputs']
outputs_test = npfile_test['outputs']

npfile_test.close()

testset_np = (inputs_test-inputMeans)/inputStdDevs
testset_tens = torch.tensor(testset_np).float()
testset = testset_tens.cuda()
testset_tens = None

mean_errors = []
mean_diffs = []
stdevs = []

for trial in range(trials):
    # put standardized input vectors through model
    results = []
    targets = []
    errors = []
    diffs = []
        
    model = torch.load('{}_{}'.format(outfile,trial))

    for i in range(len(testset)):
        # get model output
        model.eval()
        test_output_tens = model(testset[i])
        test_output_cpu = test_output_tens.cpu()
        test_output_tens = None
        test_output = test_output_cpu.detach().numpy()
        test_output_cpu = None
                
        # unstandardize output
        test_result = (test_output*outputStdDevs)+outputMeans
        test_output = None
        results.append(test_result)
                
        # get target result
        targ_vec = outputs_test[i]
        targets.append(targ_vec)
                
        # get accuracy
        diff = np.linalg.norm(targ_vec - test_result)
        mag = np.linalg.norm(targ_vec)
        error = (diff/mag)*100
        errors.append(error)
        diffs.append(diff)

    # get stats for each trial
    stdev_targ = np.std(targets)
    stdev_diff = np.std(diff)
    stdev_diff_targ = stdev_diff / stdev_targ
    stdevs.append(stdev_diff_targ)
        
    # get mean error, diff for each trial
    mean_error = np.mean(errors)
    mean_errors.append(mean_error)

    mean_diff = np.mean(diffs)
    mean_diffs.append(mean_diff)

# save results

np.save(file='{}_err'.format(outfile), arr=mean_errors)
np.save(file='{}_diff'.format(outfile), arr=mean_diffs)
np.save(file='{}_stdev'.format(outfile), arr=stdevs)
    
print('Done!')
