import torch
from torchinfo import summary
from resnetv2_32 import ResNetV2

# init random tensor

x = torch.randn(2, 3, 224, 224)

# init model & get summary

model = ResNetV2()

summary(model, input_size = x.size())
