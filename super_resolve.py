""""Application du CNN"""

from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--inputPath', type=str, required=True, help='input image to use')
parser.add_argument('--modelPath', type=str, required=True, help='model file to use')
parser.add_argument('--outputPath', type=str, default="out.png", help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

opt.cuda = True

# print(opt)

"""
try:
    os.mkdir(opt.output_filename.split('/')[0])
except FileExistsError:
    pass
"""

img = Image.open(opt.inputPath).convert("RGBA")
r, g, b, a = img.split()

rgb = [r, g, b]

model = torch.load(opt.modelPath)
img_to_tensor = ToTensor()

for i in range(len(rgb)):
    rgb[i] = img_to_tensor(rgb[i]).view(1, -1, rgb[i].size[1], rgb[i].size[0])
    rgb[i] *= 256.0

rgb = torch.cat((rgb[0], rgb[1], rgb[2]), 1)

if opt.cuda:
    model = model.cuda()
else:
    model = model.cpu()

if opt.cuda:
    input_img = rgb.cuda()
else:
    input_img = rgb.cpu()

print(input_img)

out = model(input_img)
out = out.cpu()

out_img_y = out[0].detach().numpy()
out_img_y = out_img_y[0]
out_img_y *= 256.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')

out = out_img_y

print(type(out))

out_img = Image.merge("RGB", out)

out_img.save(opt.outputPath)
print('output image saved to', opt.outputPath)
