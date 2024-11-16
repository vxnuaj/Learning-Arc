import torch
from torchinfo import summary
from densenet import DenseNet

# init randn tensor

x = torch.randn(size = (3, 3, 224, 224))

# init model

model = DenseNet( k = 40, theta = .5)

# get summary and final output shape

summary(model, x.size())

print(f"\nFinal Output Shape: {model(x).size()}")