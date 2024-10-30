import torch
import torch.nn as nn
from torchinfo import summary
from inceptionv1 import InceptionV1

# init random tensor
x = torch.randn(1, 3, 224, 224)

# init model

model = InceptionV1()

summary(model, input_data = x)
