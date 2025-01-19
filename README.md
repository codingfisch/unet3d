# unet3d
`pip install unet3d` to get single-file implementations of 3D UNet architectures for 3D image segmentation!

## Usage 💡
Run a `UNet3d`...
```python
import torch
from unet3d import UNet3d, LinkNet3d

unet = UNet3d(c_in=1, c_out=2, c_start=8, layers=4)
x = torch.rand(1, 1, 64, 64, 64)
output = unet(x)
print(output.shape)  # (1, 2, 64, 64, 64)
```
...or a `LinkNet3d`—a more compute efficient UNet variant—on `torch.rand`om data 🤓
```python
linknet = LinkNet3d(c_in=1, c_out=2, c_start=8, layers=4)
x = torch.rand(1, 1, 64, 64, 64)
output = linknet(x)
print(output.shape)  # (1, 2, 64, 64, 64)
```
