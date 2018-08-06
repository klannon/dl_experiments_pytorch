#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import copy
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(description='Train ANN auto-encoder.')
parser.add_argument('trainfile',
                        help='File name for training set')

parser.add_argument('testfile',
                        help='File name for test set')

parser.add_argument('-o','--out', dest='outbase', metavar='OUTBASE',
                        help='File name for saving model results')

parser.add_argument('-m', '--model', help='Model to test')

parser.add_argument('-t','--trials', type=int, 
                        help='Number of trials for the model')

parser.add_argument('-h1','--hl1-nodes', type=int, 
                        help='Number of nodes in the first hidden layer of the conversion network for model being tested')

parser.add_argument('-h2','--hl2-nodes', type=int, 
                        help='Number of nodes in the second hidden layer of the conversion network for model being tested. 
                            If no second layer, choose zero.')

parser.add_argument('-h3','--hl3-nodes', type=int, 
                        help='Number of nodes in the third hidden layer of the conversion network for model being tested. 
                            If no third layer, choose zero.')


args = parser.parse_args()

infile = args.trainfile
infile_test = args.testfile
seed = np.random.randint(low=0, high=1000000, size=1)

trials = args.trials
H1 = args.hl1_nodes
H2 = args.hl2_nodes
H3 = args.hl3_nodes  

outfile = args.outbase

use_cuda = torch.cuda.is_available()
print(use_cuda)

# defining model class

class SharedModel(torch.nn.Module):
    def __init__(self,H1, H2, H3):
        for H2 == H3 == 0:
          super(SharedModel,self).__init__()
          self.linear1 = torch.nn.Linear(3,H1)
          torch.nn.init.kaiming_normal_(self.linear1.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear1.bias,0)
          self.activation = torch.nn.PReLU()
          self.linear2 = torch.nn.Linear(H1,3)
          torch.nn.init.kaiming_normal_(self.linear2.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear2.bias,0)
          self.linear3 = torch.nn.Linear(6,3)
          torch.nn.init.kaiming_normal_(self.linear3.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear3.bias,0)
          self.linear4 = torch.nn.Linear(3,3)
          torch.nn.init.kaiming_normal_(self.linear4.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear4.bias,0)
                    
         for H2 > 0 and H3 == 0:
          super(SharedModel,self).__init__()
          self.linear1 = torch.nn.Linear(3,H1)
          torch.nn.init.kaiming_normal_(self.linear1.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear1.bias,0)
          self.activation = torch.nn.PReLU()
          self.linear2 = torch.nn.Linear(H1,H2)
          torch.nn.init.kaiming_normal_(self.linear2.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear2.bias,0)
          self.activation = torch.nn.PReLU()
          self.linear3 = torch.nn.Linear(H2,3)
          torch.nn.init.kaiming_normal_(self.linear3.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear3.bias,0)
          self.linear4 = torch.nn.Linear(6,3)
          torch.nn.init.kaiming_normal_(self.linear4.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear4.bias,0)
          self.linear5 = torch.nn.Linear(3,3)
          torch.nn.init.kaiming_normal_(self.linear5.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear5.bias,0)
                    
         for H2 > 0 and H3 > 0:
          super(SharedModel,self).__init__()
          self.linear1 = torch.nn.Linear(3,H1)
          torch.nn.init.kaiming_normal_(self.linear1.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear1.bias,0)
          self.activation = torch.nn.PReLU()
          self.linear2 = torch.nn.Linear(H1,H2)
          torch.nn.init.kaiming_normal_(self.linear2.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear2.bias,0)
          self.activation = torch.nn.PReLU()
          self.linear3 = torch.nn.Linear(H2,H3)
          torch.nn.init.kaiming_normal_(self.linear3.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear3.bias,0)
          self.activation = torch.nn.PReLU()
          self.linear4 = torch.nn.Linear(H3,3)
          torch.nn.init.kaiming_normal_(self.linear4.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear4.bias,0)
          self.linear5 = torch.nn.Linear(6,3)
          torch.nn.init.kaiming_normal_(self.linear5.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear5.bias,0)
          self.linear6 = torch.nn.Linear(3,3)
          torch.nn.init.kaiming_normal_(self.linear6.weight,a=0.25, nonlinearity='leaky_relu')
          torch.nn.init.constant_(self.linear6.bias,0)

    def forward(self,x):
        for H2 == H3 == 0:
          vars = torch.chunk(x,2,-1)
          conv_results = []
          for v in vars:
              o = self.linear1(v)
              o = self.activation(o)
              o = self.linear2(o)
              conv_results.append(o)
          x_cat = torch.cat(conv_results,-1)
          o = self.linear3(x_cat)
          o = self.activation(o)
          x_pred = self.linear4(o)
          return x_pred
                    
        for H2 > 0 and H3 == 0:
          vars = torch.chunk(x,2,-1)
          conv_results = []
          for v in vars:
              o = self.linear1(v)
              o = self.activation(o)
              o = self.linear2(o)
              o = self.activation(o)
              o = self.linear3(o)
              conv_results.append(o)
          x_cat = torch.cat(conv_results,-1)
          o = self.linear4(x_cat)
          o = self.activation(o)
          x_pred = self.linear5(o)
          return x_pred  
                    
        for H2 > 0 and H3 > 0:
          vars = torch.chunk(x,2,-1)
          conv_results = []
          for v in vars:
              o = self.linear1(v)
              o = self.activation(o)
              o = self.linear2(o)
              o = self.activation(o)
              o = self.linear3(o)
              o = self.activation(o)
              o = self.linear4(o)
              conv_results.append(o)
          x_cat = torch.cat(conv_results,-1)
          o = self.linear5(x_cat)
          o = self.activation(o)
          x_pred = self.linear6(o)
          return x_pred  


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

for trial in range(trials):
    # put standardized input vectors through model
    results = []
    targets = []
    errors = []
    diffs = []
        
    model = torch.load(args.model)
   
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

    # get mean error, diff for each trial
    mean_error = np.mean(errors)
    mean_errors.append(mean_error)

    mean_diff = np.mean(diffs)
    mean_diffs.append(mean_diff)

# save results

np.save(file='{}_err'.format(outfile), arr=mean_errors)
np.save(file='{}_diff'.format(outfile), arr=mean_diffs)
    
print('Done!')
