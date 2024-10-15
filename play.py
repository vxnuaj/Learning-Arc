import torch
import torch.nn.functional as F

x = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype = torch.float32)
x_interpolated = F.interpolate(x, size = (5, 5), mode = 'bilinear')


print(f"Data:\n\n{x}\n\n")
print(f"Data Bilinearly Interpolated:\n\n{x_interpolated}\n\n")

y = torch.tensor([[[[1, 2], [3, 4]]]], dtype = torch.float32)
y_interpolated = F.interpolate(y, size=(3, 3), mode = 'nearest')

print(y_interpolated)