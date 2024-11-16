import torch
from torchinfo import summary
from SEResNet import SEResNet

# init randn tensor

x = torch.randn( size = (2, 3, 224, 224))

# init model

model = SEResNet( reduct_ratio = 16) # usign recommended reduction ratio | https://arxiv.org/pdf/1709.01507

# get model summary and final shape

summary(model, x.size())
