import torch
from torchinfo import summary
from nin import NiN

# init model
model = NiN()

# init random tensor
x = torch.randn(size = (1, 3, 224, 224))

summary(model, input_size = x.size())