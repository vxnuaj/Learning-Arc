import torch
from torchinfo import summary
from xception import Xception

# dummy randn tensor

x = torch.randn(size = (2, 3, 299, 299))

# init model

model = Xception( fc = True ) # including optional fc layers

# get model summary and final output shape

summary(model, x.size())
print(f"\nFinal Output Size: {model(x).size()}")

