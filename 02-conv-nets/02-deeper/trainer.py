import torch
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
      
    @staticmethod 
    def train_test(model, X_train, Y_train, X_test, Y_test, loss_fn, optimizer, epochs, device, batch_size = 64, seed = 1):
     
        torch.manual_seed(seed)
      
        model.to(device) 
        model.train()
      
        dataset = TensorDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True) 
        
        print(f"Training {type(model).__name__} for {epochs} epochs on {device}.")
        
        for epoch in range(epochs):
            
            i = 0 
            
            for X_batch, Y_batch in dataloader:
                
                i += 1
               
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                
                logits = model(X_batch) 
                pred = logits.argmax(dim = 1)
                loss = loss_fn(logits, Y_batch)
                acc = Trainer._accuracy(Y_batch, pred)
                
                print(f"Epoch: {epoch} | Iteration: {i} | Loss: {loss} | Accuracy: {acc}")
            
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() 
         
        with torch.no_grad():     
           
            X_test = X_test.to(device)
            Y_test = Y_test.to(device) 
            
            logits = model()
            pred = logits.argmax(dim = 1)
            loss = loss_fn(logits, Y_test)
            acc = Trainer._accuracy(Y_test, pred)
            
            print(f"Testing | Loss: {loss} | Accuracy: {acc}")

    @staticmethod
    def _accuracy(Y, pred):
        return (Y == pred).sum().item() / Y.size(dim = 0) * 100