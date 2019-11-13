###This is the pyfile for training###
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


embedding_size = 512
hidden_size = 512
num_layers = 2
rnn_dropout = 0.1

##The 'dictionary' of the all the audio frequency data##
audio_dict = pkl.load(open('audio_dict.pkl',"rb"))

##split the data into training and test set by the ratio 8:2 in chronolgical order.
data_list = []
for root, dirs, files in os.walk('/scratch/jq689'):
    for fname in files:
        if fname.endswith('.json'):
            if fname.endswith('mfcc.json'):
                continue
            else:
                path=os.path.join(root,fname)
                data_list.append(path)
data_year = [f.split('/')[-1].split('_')[0] for f in data_list]
data_sorted = [x for _,x in sorted(zip(data_year,data_list))]
data_len = len(data_sorted)
train_per = 0.8
train_path = data_sorted[:int(data_len * train_per)]
# val = data_sorted[int(data_len * train_per):] 
test_path = data_sorted[int(data_len * train_per):]


###all the options of the model###
options = {
    'num_embeddings': len(audio_dict),
    'embedding_dim': embedding_size,
    'input_size': embedding_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'rnn_dropout': rnn_dropout,
    'num_classes': 3
}

###This model contains three main layers: the first is an embedding layer, the second is the LSTM model, 
###and the third one is a projection layer.

class LSTMLanguageModel(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.lookup = nn.Embedding(num_embeddings=options['num_embeddings'], embedding_dim=options['embedding_dim'])
        self.lstm = nn.LSTM(options['input_size'], options['hidden_size'], options['num_layers'], batch_first=True)
        self.projection = nn.Linear(options['hidden_size'], options['num_classes'])
    def forward(self, encoded_input_sequence):
        embeddings = self.lookup(encoded_input_sequence)
        lstm_outputs = self.lstm(inp)
        logits= self.projection(lstm_outputs[0])
        return logits

###Choose cross entrophy loss and 'Adam' optimizer for now for the LSTM model
model = LSTMLanguageModel(options).cuda()
criterion = nn.CrossEntropyLoss()
model_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(model_parameters, lr=0.001)

##number of epoch to train
number_of_epoch = 1

for epoch_number in range(number_of_epoch):
    model.train()
    for path in train_path:
        pickle_in = open(path,"rb")
        file_dict = pkl.load(pickle_in)
        inp = torch.Tensor(file_dict['freq']).long().cuda()
        target = torch.Tensor(file_dict['freq'][1:]).long().cuda()
        logits = model(inp)#####This is the step where the HPC cluster told me there is not enough memeory##
        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()


model.eval()
##evaluate the model using test data by using accuracy
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
##save the model
torch.save(model.state_dict(), "LSTM_baseline_model.ckpt")




    