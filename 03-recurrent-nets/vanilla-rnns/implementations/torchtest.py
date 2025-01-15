import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchRNN import RNN



# load data

X = np.loadtxt('data/sequences.csv', delimiter = ',')
Y = np.loadtxt('data/targets.csv', delimiter = ',')

X_train = X[0:-1000]
Y_train = Y[0:-1000]
X_test = X[-1000:]
Y_test = Y[-1000:]

X_train = torch.from_numpy(X_train).to(dtype = torch.int64)
Y_train = torch.from_numpy(Y_train).to(dtype = torch.int64)
X_test = torch.from_numpy(X_test).to(dtype = torch.int64)
Y_test = torch.from_numpy(Y_test).to(dtype = torch.int64)

rnn = RNN()

print(rnn(X_train[0:2]).shape)

'''# minibatching
class SequenceDataset(Dataset):
    
    def __init__(self, sequences, labels):
        
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

dataset = SequenceDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size = 1024, shuffle = True)

# init model and init training 

device = ('mps')
rnn = RNN().to(device)
epochs = 100
lr = .1

optim = opt.Adam(rnn.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
rnn.train()


print(f"Training")


for epoch in range(epochs):
    for iteration, batch in enumerate(dataloader):
        seq = batch[0].to(device)
        labels = batch[1].to(device)
       
        logits = rnn(seq) 
        loss = criterion(logits.view(-1, 26), labels.view(-1))

        if iteration % 10 == 0:
            print(f"Epoch: {epoch} | Iteration: {iteration} | Loss: {loss.item()}")

        loss.backward()
        optim.step()
        optim.zero_grad()
        '''