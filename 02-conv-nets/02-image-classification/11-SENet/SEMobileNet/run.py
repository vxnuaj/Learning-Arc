import torch
from torchinfo import summary
from SEMobileNet import SEMobileNetV1

# init model -- res mult = .5, depth mult = .75, as example

model = SEMobileNetV1(rho = .5, alpha = .75)

# init randn tensor

x = torch.randn( size = (2, 3, 224, 224))

# run model, get summary, and final output size

summary(model, x.size())
print(f"\nFinal  Output Size: {model(x).size()}")
