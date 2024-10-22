import torch
from torchinfo import summary
from vgg import VGG16

# init random shape
x = torch.randn(1, 3, 224, 224)

# init model and run a forward pass.
model = VGG16()
y = model.forward(x)

summary(model, input_size = x.size()) 
