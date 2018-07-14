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

parser = argparse.ArgumentParser(description='Train ANN auto-encoder.')
parser.add_argument('trainfile',
                        help='File name for training set')

parser.add_argument('testfile',
                        help='File name for test set')

parser.add_argument('-o','--out', dest='outbase', metavar='OUTBASE',
                        help='File name base (no extension)')
    
parser.add_argument('-s','--start-node',
                        type=int, help='Number of hidden layer nodes to start with during model generation')

parser.add_argument('-e','--end-node',
                        type=int, help='Number of hidden layer nodes to stop at in model generation')

parser.add_argument('-i','--node-interval',
                        default=1, type=int, help='Step size for number of hidden layer nodes in model generation')

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

results_file =outfile_name = '{}_{}to{}nodes_step{}'.format(args.outbase,args.start_node,args.end_node,args.node_interval)

infile = args.trainfile
infile_test = args.testfile
seed = np.random.randint(low=0, high=1000000, size=1)

nodes = np.arange(args.start_node, args.end_node + args.node_interval, args.node_interval)
num_models = len(nodes)
trials = args.trials
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
inputs = (inputs-inputMeans)/inputStdDevs
inputs = torch.from_numpy(inputs).float()
inputs = inputs.cuda()

outputMeans = outputs.mean(axis=0)
outputStdDevs = outputs.std(axis=0)
outputs = (outputs-outputMeans)/outputStdDevs
outputs = torch.from_numpy(outputs).float()
outputs = outputs.cuda()

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
                best_epoch = epoch[0]
                    
    # load and save the best model
    torch.save(best_model, modelpath)
        
    return

for node in nodes:
    network = OrderedDict([])

    layer_1 = nn.Linear(int(3), int(node))
    torch.nn.init.kaiming_uniform_(layer_1.weight, a=0.25, nonlinearity='leaky_relu')
    torch.nn.init.constant_(layer_1.bias,0)

    network = OrderedDict([('lin1', layer_1),('relu1', nn.PReLU())]) 

    layer_2 = nn.Linear(int(node), int(3))  
    torch.nn.init.kaiming_uniform_(layer_2.weight, a=0.25, nonlinearity='leaky_relu')
    torch.nn.init.constant_(layer_2.bias,0)

    network['lin2'] = layer_2  

    model = nn.Sequential(network)

    model = model.cuda()

    for trial in range(trials):
        run_training(model = model, modelpath = 'v_conv_model_{}node_{}'.format(node,trial), trainset = train_set, validsize = valid_size, numepochs = n_epochs, batchsize = batch_size, seed = seed)
        print('Training complete: {} node model, trial {}'.format(node,trial))


## Testing the models

npfile_test = np.load(infile_test)
inputs_test = npfile_test['inputs']
outputs_test = npfile_test['outputs']

npfile_test.close()

testset_np = (inputs_test-inputMeans)/inputStdDevs
testset = torch.tensor(testset_np).float()
testset = testset.cuda()

model_errors = np.empty([num_models,2])
    
for node in nodes:

    mean_errors = []

    for trial in range(trials):
        # put standardized input vectors through model
        results = []
        targets = []
        errors = []

        for i in range(len(testset)):
            # get model output
            model.eval()
            test_output = model(testset[i])
            test_output = test_output.cpu()
            test_output = test_output.detach().numpy()
                
            # unstandardize output
            test_result = (test_output*outputStdDevs)+outputMeans
            results.append(test_result)
                
            # get target result
            targ_vec = outputs_test[i]
            targets.append(targ_vec)
                
            # get accuracy
            diff = np.linalg.norm(targ_vec - test_result)
            mag = np.linalg.norm(targ_vec)
            error = (diff/mag)*100
            errors.append(error)

        # get mean error for each trial
        mean_error = np.mean(errors)
        mean_errors.append(mean_error)

    # get mean error for each model
    model_error = np.mean(mean_errors)
    for i in range(len(nodes)):
        model_errors[i][0] = node
        model_errors[i][1] = model_error

# sort models from lowest to highest mean error, save five best models
model_errors_ascending = model_errors[np.argsort(model_errors[:, 1])]
np.savetxt(fname=results_file, X=model_errors_ascending)
    
print('Done!')



        
            


