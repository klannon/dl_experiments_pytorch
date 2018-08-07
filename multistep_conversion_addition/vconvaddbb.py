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

parser.add_argument('testfile',
                        help='File name for test set')

parser.add_argument('-o','--out', dest='outbase', metavar='OUTBASE',
                        help='File name base for model and results (trial number will be appended)')

parser.add_argument('-l1','--layer1-nodes',
                        type=int, help='Number of nodes in the first hidden layer')
    
parser.add_argument('-l2','--layer2-nodes',
                        type=int, help='Number of nodes in the second hidden layer of the conversion network. 
                            If only one layer wanted, choose zero.')

parser.add_argument('-b','--batch-size',
                        default=250, type=int, help='Batch size for data during training and testing')

parser.add_argument('-n','--num-epochs', type=int, 
                        help='Number of epochs in a training cycle')

parser.add_argument('-v','--valid-size', type=int, 
                        help='Size of the validation set')

parser.add_argument('-t','--trials', type=int, 
                        help='Number of trials for the model')                        

args = parser.parse_args()

## User Inputs

infile = args.trainfile
infile_test = args.testfile
seed = np.random.randint(low=0, high=1000000, size=1)

batch_size = args.batch_size
valid_size = args.valid_size
n_epochs = args.num_epochs
hl1_nodes = args.layer1_nodes
hl2_nodes = args.layer2_nodes
hl3_nodes = 6
hl4_nodes = 3
trials = args.trials

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

# creating the model

network = OrderedDict([])

layer_1 = nn.Linear(int(6), int(hl1_nodes))
torch.nn.init.kaiming_uniform_(layer_1.weight, a=0.25, nonlinearity='leaky_relu')
torch.nn.init.constant_(layer_1.bias,0)

network = OrderedDict([('lin1', layer_1),('relu1', nn.PReLU())]) 

layer_2 = nn.Linear(int(hl1_nodes), int(hl2_nodes))  
torch.nn.init.kaiming_uniform_(layer_2.weight, a=0.25, nonlinearity='leaky_relu')
torch.nn.init.constant_(layer_2.bias,0)

network['lin2'] = layer_2  

layer_3 = nn.Linear(int(hl2_nodes), int(6))  
torch.nn.init.kaiming_uniform_(layer_3.weight, a=0.25, nonlinearity='leaky_relu')
torch.nn.init.constant_(layer_3.bias,0)

network['lin3'] = layer_3  

layer_4 = nn.Linear(int(6), int(3))  
torch.nn.init.kaiming_uniform_(layer_4.weight, a=0.25, nonlinearity='leaky_relu')
torch.nn.init.constant_(layer_4.bias,0)

network['lin4'] = layer_4  

layer_5 = nn.Linear(int(3), int(3))  
torch.nn.init.kaiming_uniform_(layer_5.weight, a=0.25, nonlinearity='leaky_relu')
torch.nn.init.constant_(layer_5.bias,0)

network['lin5'] = layer_5  

model = nn.Sequential(network)
model = model.cuda()

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


# runs training and validation process over all epochs and then tests each model
def run_training_testing(model, modelpath, trainset, validsize, numepochs, batchsize, seed):
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

    # put standardized input vectors through model
    results = []
    targets = []
    errors = []
    diffs = []
        
    model = torch.load(modelpath)
   
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

    # save results
    np.save(file='{}_err_{}'.format(outfile,trial), arr=mean_errors)
    np.save(file='{}_diff_{}'.format(outfile,trial), arr=mean_diffs)
    
print('Done!')

## Run Training and Testing Function

for trial in range(trials):
    
    run_training_testing(model=model, modelpath='{}_{}'.format(outfile,trial), trainset=train_set, validsize=valid_size, 
                 numepochs=n_epochs, batchsize=batch_size, seed=seed)

print('Finished!')