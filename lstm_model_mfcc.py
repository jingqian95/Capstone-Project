####Very similar to the lstm model for the frequency data, the difference is that there is no
###embedding layer here. Our mfcc data has the shape torch.Size([399094, 20]). 
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
import sys



hidden_size = 256
num_layers = 2
rnn_dropout = 0.1

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

options = {
    'input_size': 399094,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'rnn_dropout': rnn_dropout,
    'num_classes': 3
}

class LSTMLanguageModel(nn.Module):
    def __init__(self, options):
        super().__init__()
        #self.lookup = nn.Embedding(num_embeddings=options['num_embeddings'], embedding_dim=options['embedding_dim'])
        self.lstm = nn.LSTM(options['input_size'], options['hidden_size'], options['num_layers'], batch_first=True)
        self.projection = nn.Linear(options['hidden_size'], options['num_classes'])
    def forward(self, encoded_input_sequence):
        lstm_outputs = self.lstm(encoded_input_sequence)
        logits= self.projection(lstm_outputs[0])
        return logits


model = LSTMLanguageModel(options).cuda()
criterion = nn.CrossEntropyLoss()
model_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(model_parameters, lr=0.001)

number_of_epoch = 1

for epoch_number in range(number_of_epoch):
    model.train()
    for path in train_path:
        pickle_in = open(path,"rb")
        file_dict = pkl.load(pickle_in)
        inp = torch.Tensor(file_dict['freq']).long()
        target = torch.Tensor(file_dict['freq'][1:]).long()
        logits = model(inp)
        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()


model.eval()
total=0
correct=0
with torch.no_grad():
    for path in test_path:
        pickle_in = open(path,"rb")
        file_dict = pkl.load(pickle_in)
        inp = torch.Tensor(file_dict['freq']).long()
        target = torch.Tensor(file_dict['freq'][1:]).long()
        logits = model(inp)
        outputs = F.softmax(logits,dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += target.size(0)
        correct += predicted.eq(target.view_as(predicted)).sum().item()
print((100 * correct / total))
torch.save(model.state_dict(), "LSTM_baseline_model.ckpt")



