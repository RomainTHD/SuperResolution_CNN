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

img = Image.open(opt.inputPath).convert('YCbCr')
y, cb, cr = img.split()

model = torch.load(opt.modelPath)
img_to_tensor = ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

if opt.cuda:
    model = model.cuda()
    input = input.cuda()

out = model(input)
out = out.cpu()
out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img.save(opt.outputPath)
print('output image saved to', opt.outputPath)
