import torch
from torchinfo import summary
from resnet32 import ResNet

# init random tensor

x = torch.randn(2, 3, 224, 224)

# init model & get summary

model = ResNet()

summary(model, input_size = x.size())
