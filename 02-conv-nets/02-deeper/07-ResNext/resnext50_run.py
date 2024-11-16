import torch
from torchinfo import summary
from resnext50 import ResNext50

# init randn tensor

x = torch.randn(size = (2, 3, 224, 224))

# init model & get summary

model = ResNext50()

summary(model, input_size = x.size())
