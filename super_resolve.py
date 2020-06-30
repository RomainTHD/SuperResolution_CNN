# -*- coding: utf-8 -*-
"""Application du CNN sur une image"""

########################################################################################################################
# Gestion Python 2.x

from __future__ import print_function
from __future__ import unicode_literals

import sys

if sys.version_info < (3, 0):
    print("Nécessite Python 3.x !")
    sys.exit(1)

########################################################################################################################
# Modules et paramètres

try:
    import torch
except ImportError:
    print("Nécessite PyTorch !")
    sys.exit(1)

import argparse
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

# Paramètres
parser = argparse.ArgumentParser(description="Super-résolution, application du modèle\n\nBIEN LIRE LE README")
parser.add_argument('-i', '--inputPath', type=str, required=True, help="Image d'entrée")
parser.add_argument('-m', '--modelPath', type=str, required=True, help="Modèle .pth à utiliser")
parser.add_argument('-o', '--outputPath', type=str, default="out.png", help="Image de sortie")
parser.add_argument('--cpu', action='store_true', help="Utilise le CPU et non le GPU / CUDA")
parser.add_argument('-q', '--quiet', action='store_true', help="Silencieux")
opt = parser.parse_args()

if not opt.quiet:
    print("Paramètres :")

    for arg in vars(opt):
        print(arg, "->", getattr(opt, arg))

if not opt.cpu and not torch.cuda.is_available():
    raise Exception("Erreur CUDA : pas de GPU trouvée. Utilisez --cpu")

########################################################################################################################
# Super-résolution

# On divise l'image en 4 channels (seuls r, g et b seront utilisés)
img = Image.open(opt.inputPath).convert("RGBA")
r, g, b, a = img.split()

# Chargement du modèle
model = torch.load(opt.modelPath)

# Tenseur PyTorch
img_to_tensor = ToTensor()

# Transforme les 3 channels en tenseurs
input_img_all = [img_to_tensor(x).view(1, -1, x.size[1], x.size[0]) for x in (r, g, b)]

# Conversion du modèle selon si CUDA ou non
if opt.cpu:
    model = model.cpu()
else:
    model = model.cuda()

# Conversion des tenseurs d'image selon si CUDA ou non
for i in range(len(input_img_all)):
    if opt.cpu:
        input_img_all[i] = input_img_all[i].cpu()
    else:
        input_img_all[i] = input_img_all[i].cuda()

# Les 3 channels
out_all = []

for i in range(3):
    # Application du modèle
    out = model(input_img_all[i])

    # Transformation vers CPU pour l'utiliser avec numpy
    out = out.cpu()

    # Transformation en tableau numpy
    out_img_y = out[0].detach().numpy()
    out_img_y = out_img_y[0]

    # [0, 1[ -> [0, 256[
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)

    # Tableau numpy -> tableau classique
    out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')

    out_all.append(out_img_y)

########################################################################################################################
# Fusion des channels
out_img = Image.merge("RGB", out_all)
out_img.save(opt.outputPath)

if not opt.quiet:
    print("Image sauvegardée vers", opt.outputPath)
