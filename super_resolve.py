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
parser.add_argument('--modelPathR', type=str, required=True, help='model file to use')
parser.add_argument('--modelPathG', type=str, required=True, help='model file to use')
parser.add_argument('--modelPathB', type=str, required=True, help='model file to use')
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

model_all = [torch.load(x) for x in (opt.modelPathR, opt.modelPathR, opt.modelPathR)]

img_to_tensor = ToTensor()

input_img_all = [img_to_tensor(x).view(1, -1, x.size[1], x.size[0]) for x in (r, g, b)]

for i in range(len(model_all)):
    if opt.cuda:
        model_all[i] = model_all[i].cuda()
    else:
        model_all[i] = model_all[i].cpu()

for i in range(len(input_img_all)):
    if opt.cuda:
        input_img_all[i] = input_img_all[i].cuda()
    else:
        input_img_all[i] = input_img_all[i].cpu()

out_all = []

for i in range(len(model_all)):
    out = (model_all[i])(input_img_all[i])
    out = out.cpu()

    out_img_y = out[0].detach().numpy()
    out_img_y = out_img_y[0]

    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)

    out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')

    out_all.append(out_img_y)

out_img = Image.merge("RGB", out_all)

out_img.save(opt.outputPath)
print('output image saved to', opt.outputPath)
