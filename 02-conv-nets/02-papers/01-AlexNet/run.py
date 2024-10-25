import torch
import torch.nn as nn
import torch.optim as opt
import cv2
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from torchvision import transforms
from PIL import Image
from alexnet import AlexNet

# init random shape
x = torch.randn(1, 3, 224, 224)

# init model and run a forward pass.
model = AlexNet()
y = model.forward(x)

summary(model, input_size = x.size()) # correct