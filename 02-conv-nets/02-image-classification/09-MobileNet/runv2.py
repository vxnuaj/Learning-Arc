import torch
from torchinfo import summary
from mobilenetv2 import MobileNetV2

# init randn tensor

x = torch.randn(size = (2, 3, 224, 224))

# init. model

model = MobileNetV2(alpha = 1, rho = 1)

# get model summary & final output size

summary(model, x.size())
print(f'\nFinal Output Size: {model(x).size()}')
