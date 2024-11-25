import torch
from torchinfo import summary
from squeezenet import SqueezeNet


# init randn tensor

x = torch.randn( size = (2, 3, 224, 224))

# init model

model = SqueezeNet(

    base_e = 128, 
    incr_e = 128,
    pct_3x3 = .5,
    freq = 2,
    sr = .125

)

# run model, get summary, get final output shape

summary(model, x.size())
print(f"\nFinal Model Size: {model(x).size()}")
