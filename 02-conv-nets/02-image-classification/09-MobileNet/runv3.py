import torch
from torchinfo import summary
from mobilenetv3 import MobileNetV3Large

# init randn tensor

x = torch.randn(size = (2, 3, 224, 224))

# init model

model = MobileNetV3Large( k = 1000 ) # for 1k imagenet classes

# get model summary and final output shape

summary(model, x.size())
print(f"\nFinal Output Shape: {model(x).size()}")
