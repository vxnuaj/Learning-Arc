import torch
import torch.nn as nn
import torch.optim as opt
from mnist1d.data import make_dataset, get_dataset_args

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            
        nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 5), nn.LeakyReLU(), nn.MaxPool1d(kernel_size = 2),  # -> 18 per channel
        nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 3), nn.LeakyReLU(), nn.MaxPool1d(kernel_size = 2), # -> 8 per channel
        nn.Flatten(),
        nn.Linear(in_features=32 * 8, out_features=64), nn.LeakyReLU(),
        nn.Linear(in_features=64, out_features= 10) 
        
        )
       
    def forward(self, x):
        return self.model(x) 
        
    def accuracy(self, pred, y):
        return (pred == y).sum().item() / y.size(0)
        
if __name__ == "__main__":
   
    torch.manual_seed(1)
    device = ('mps')
    
    defaults = get_dataset_args()
    data = make_dataset(defaults)
    x, y = data['x'].reshape(4000, 1, 40), data['y']
    
    x = torch.tensor(x, device = device, dtype = torch.float32) / 255
    y = torch.tensor(y, device= device, dtype = torch.float32)
    
    model = NN().to(device)
   
    lr = .01
    betas = (.9, .99)  
    epochs = 1000
    optim = opt.Adam(model.parameters(), lr, betas = betas) 
    criterion = nn.CrossEntropyLoss()
   
    model.train()
    
    for epoch in range(epochs):
        logits = model(x)
        pred = logits.argmax(dim = 1)
        loss = criterion(logits, y)
        acc = model.accuracy(pred, y)
        
        print(f"Epoch: {epoch + 1} | Accuracy: {acc} | Loss: {loss}")
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        