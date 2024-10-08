import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from nue.preprocessing import csv_to_numpy, x_y_split

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5), nn.Tanh(), # -> 6 x 28 x 28
            nn.AvgPool2d(kernel_size = 2, stride = 2), # -> 6 x 14 x 14
            nn.Conv2d(in_channels = 6, out_channels=16, kernel_size=5), nn.Tanh(), # -> 16 x 10 x 10
            nn.AvgPool2d(kernel_size = 2, stride = 2), # -> 16 x 5 x 5
            nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5), nn.Tanh(), # -> 120 x 1 x 1
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
            
        )
        
    def forward(self, x):
        return self.model(x) 

    def accuracy(self, pred, y):
        return (pred == y).sum().item() / y.size(0)

if __name__ == "__main__":

    device = ('mps')

    print(f"Preprocessing")
    train = csv_to_numpy('data/mnist_train.csv')
    test = csv_to_numpy('data/mnist_test.csv')
    X_train, Y_train = x_y_split(train, y_col = 'first')
    X_test, Y_test = x_y_split(test, y_col = 'first')
    X_train = X_train / 255
    X_test = X_test / 255
    X_train = torch.tensor(X_train, dtype = torch.float32, device = device).reshape((60000, 1, 28, 28))
    Y_train = torch.tensor(Y_train, dtype = torch.float32, device = device).flatten()
    X_test = torch.tensor(X_test, dtype = torch.float32, device = device).reshape((10000, 1, 28, 28))
    Y_test = torch.tensor(Y_test, dtype = torch.float32, device = device).flatten()


    print(f"Interpolating")
    X_train = F.interpolate(X_train, size = (32, 32), mode = 'bilinear')
    X_test = F.interpolate(X_test, size = (32, 32), mode = 'bilinear')
    print(f"Instantiating Model")
    model = LeNet().to(device)
  
    epochs = 100
    lr = .1
   
    print(f"Instantiating Optims and Criterions")
    optim = opt.SGD(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
     
    print(f"Training LeNet")
    print() 
    model.train()
    
    for epoch in range(epochs): 
        logits = model(X_train)
        pred = logits.argmax(dim = 1)
        loss = criterion(logits, Y_train)
        accuracy = model.accuracy(pred, Y_train)
        
        print(f"Epoch: {epoch} | Loss: {loss} | Acc: {accuracy}")
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        
    model.eval()
    
    with torch.no_grad():
        logits = model(X_test)
        pred = logits.argmax(dim = 1)
        loss = criterion(logits, Y_train)
        accuracy = model.accuracy(pred, Y_train)
        
    print() 
    print(f"TESTING | Acc: {accuracy} | Loss: {loss}")