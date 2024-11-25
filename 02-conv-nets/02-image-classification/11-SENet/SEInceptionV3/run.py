import torch
from torchinfo import summary
from SEInceptionV3 import SEInceptionV3

# init randn tensor

x = torch.randn( size = (2, 3, 299, 299))

# init model

model = SEInceptionV3( reduct_ratio = 16 ) # usign recommended reduction ratio | https://arxiv.org/pdf/1709.01507

# get model summary and final shape

summary(model, x.size())
