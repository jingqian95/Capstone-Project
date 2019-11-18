
from pylab import*
import os
import contextlib
import pandas as pd
import numpy as np
import gzip
import pickle as pkl
import platform
import torch
import torch.nn as nn
from torch.nn import RNNCell
from torch.nn import RNNBase, RNN
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import sys

#####get the train test data split (8:2)########
data_list = []
for root, dirs, files in os.walk('/scratch/jq689'):
    for fname in files:
        if fname.endswith('mfcc.json'):
            path=os.path.join(root,fname)
            data_list.append(path)
data_year = [f.split('/')[-1].split('_')[0] for f in data_list]
data_sorted = [x for _,x in sorted(zip(data_year,data_list))]
data_len = len(data_sorted)
train_per = 0.8
train_path = data_sorted[:int(data_len * train_per)]
test_path = data_sorted[int(data_len * train_per):]


####one_hot encoding is for the target labels.
def one_hot_encoding(target,num_classes = 3,classes = [1,0,-1]):
    target_one_hot = np.zeros((target.shape[0], num_classes))
    for i, s in enumerate(target):
        idx = classes.index(s)
        target_one_hot[i, idx] = 1
    return target_one_hot

####LSTM Model######
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=8, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    def forward(self, input):
        lstm_out, hidden = self.lstm(input)
        logits = self.linear(lstm_out[-1])
        output = F.log_softmax(logits, dim=1)
        return output
    def get_accuracy(self, logits, target):
        corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()

###
num_classes = 3
current_device = 'cuda'
model = LSTM(input_dim=20, hidden_dim=128, batch_size=1, output_dim=3, num_layers=2).to(current_device)
loss_function = nn.NLLLoss()
model_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(model_parameters, lr=0.001)

####Begin the training process
model.zero_grad()
batch_size = 1
num_batches = len(train_path)/batch_size
num_epochs = 1
for epoch in range(num_epochs):
    train_running_loss, train_acc = 0.0, 0.0
    model.hidden = model.init_hidden()
    for path in train_path:
        model.zero_grad()
        pickle_in = open(path,"rb")
        file_dict = pkl.load(pickle_in)
        inp = torch.Tensor(file_dict['MFCC']).float()
        inp = inp.reshape([1,399094,-1])
        inp = inp.permute(1, 0, 2).to(current_device)####reshape the inp data into the size( sequence len, batch size, num_of_features(here is 20))
        target = one_hot_encoding(np.array([float(file_dict['label'])]))
        target = torch.max(torch.tensor(target).float(), 1)[1].to(current_device)
        logits = model(inp)
        loss = loss_function(logits, target)
        loss.backward()                                  
        optimizer.step() 
        train_running_loss += loss.detach().item()
        train_acc += model.get_accuracy(logits, target)
    print("Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f" % (epoch, train_running_loss / num_batches, train_acc / num_batches))
torch.save(model.state_dict(), "/scratch/jq689/LSTM_mfcc_baseline_model.ckpt")


####begin the evaluation#####
model.eval()
val_batches = len(test_path)/batch_size
val_loss_list, val_accuracy_list, = [], []
val_running_loss, val_acc = 0.0, 0.0
with torch.no_grad():
    for path in test_path:
        pickle_in = open(path,"rb")
        file_dict = pkl.load(pickle_in)
        inp = torch.Tensor(file_dict['MFCC']).float()
        inp = inp.reshape([1,399094,-1])
        inp = inp.permute(1, 0, 2).to(current_device)
        target = one_hot_encoding(np.array([float(file_dict['label'])]))
        target = torch.max(torch.tensor(target).float(), 1)[1].to(current_device)
        logits = model(inp)
        print('target is',target)
        print('logits are',logits)
        val_loss = loss_function(logits, target)
        val_running_loss += val_loss.detach().item()
        val_acc+= model.get_accuracy(logits, target)
        print('validation accuracy is', val_acc)
        print('validation loss is',val_loss)
val_accuracy_list.append(val_acc / val_batches)
val_loss_list.append(val_running_loss / val_batches)


