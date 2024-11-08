import torch
from torchinfo import summary
from densenet import DenseNet

# init randn tensor

x = torch.randn(size = (3, 3, 224, 224))

# init model

model = DenseNet()

# get summary and final output shape

summary(model, x.size())

print(f"\nFinal Output Shape: {model(x).size()}")