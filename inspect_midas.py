import numpy as np
from PIL import Image
import torch

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform
img = Image.new("RGB", (640, 480))
arr = np.asarray(img, dtype="float32")
res = transform(arr)
print("transform result type", type(res))
if isinstance(res, dict):
    inp = res["image"]
else:
    inp = res
print("input shape", inp.shape)
with torch.no_grad():
    depth = midas(inp)
print("depth shape", depth.shape)
