import torch
from torchinfo import summary
from inceptionv3 import InceptionV3

# init model
model = InceptionV3()

# init dummy tensor
x = torch.randn((2, 3, 299, 299))

# output & summary
summary(model, input_size = x.size())

print('holy shit this is a bigass model')