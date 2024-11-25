import torch
from torchinfo import summary
from resnetv1_32 import ResNetV1

# init random tensor

x = torch.randn(2, 3, 224, 224)

# init model & get summary

model = ResNetV1()

summary(model, input_size = x.size())
