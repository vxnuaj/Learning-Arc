import torch
from torchinfo import summary
from densenet import DenseNet

# init randn tensor

x = torch.randn(size = (3, 3, 224, 224))

# init model

model = DenseNet()

# get summary

summary(model, x.size())