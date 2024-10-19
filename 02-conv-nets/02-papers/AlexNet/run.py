import torch
from alexnet import AlexNet

# init random shape
x = torch.randn(1, 3, 224, 224)

# init model and run a forward pass.
model = AlexNet()
y = model.forward(x)

print(f"AlexNet Output Shape: {y.size()}")