import torch
import torch.nn as nn
import torch.optim as opt
from nue.preprocessing import csv_to_numpy, x_y_split


class NN(nn.Module):
    def __init__(self):
        
        super().__init__() 
        self.flatten = nn.Flatten()
        self.architecture = nn.Sequential(
            
            nn.Linear(784, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 10)
            
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.architecture(x)
        return logits
    
    def accuracy(self, pred, y):
        return (pred == y).sum().item() / y.size(dim = 0)
    
    
if __name__ == "__main__":
    torch.manual_seed(1)
    
    device = (
        'mps' if torch.backends.mps.is_available()
        else 'cpu'
    ) 

    train_data = csv_to_numpy('data/fashion-mnist_train.csv')
    X_train, Y_train = x_y_split(train_data, y_col = 'first') 
    X_train = X_train / 255
    X_train = torch.tensor(X_train, dtype = torch.float32, device = device)
    Y_train = torch.tensor(Y_train, dtype = torch.float32, device = device).flatten()
    
    test_data = csv_to_numpy('data/fashion-mnist_train.csv')
    X_test, Y_test = x_y_split(test_data, y_col = 'first')
    X_test = X_test / 255

    X_test = torch.tensor(X_test, dtype = torch.float32, device= device)
    Y_test = torch.tensor(Y_test, dtype = torch.float32, device= device).flatten()
   
    alpha = .01
    betas = (.9, .99)
    lambd = .1
    epochs = 100
   
    model = NN().to(device)
    loss_fn = nn.CrossEntropyLoss() 
    optim = opt.NAdam(model.parameters(), lr = alpha, betas = betas)
    
    print(f"USING {device.upper()}.\n")   
    for epoch in range(epochs):
        
        logits = model(X_train) 
        pred = logits.argmax(dim = 1)
        acc = model.accuracy(pred, Y_train)
        loss = loss_fn(logits, Y_train) 
        
        print(f"Epoch: {epoch + 1} | Accuracy: {acc} | Loss: {loss}")
        
        loss.backward()
        optim.step()
        optim.zero_grad()
    
        
    with torch.no_grad():
        
        logits = model.forward(X_test)
        pred = logits.argmax(dim = 1)
        acc = model.accuracy(pred, Y_test)
        loss = loss_fn(logits, Y_test)
        
        print(f"Testing Acc: {acc} | Testing Loss: {loss}")
    