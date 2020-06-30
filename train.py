"""Entrainement du CNN"""

from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_test_set

import os
import shutil
import random


def main(opt, save_dir, r=False, g=False, b=False):
    if r:
        letter = "r"
    elif g:
        letter = "g"
    elif b:
        letter = "b"
    else:
        raise AttributeError("Pas d'option sélectionnée")

    f = open(save_dir + "/args_{}.cfg".format(letter), 'w')
    print(opt, file=f)
    f.close()

    if opt.seed == 0:
        opt.seed = random.randrange(10**9)

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    device = torch.device("cpu" if opt.cpu else "cuda")

    print('===> Loading datasets')
    train_set = get_training_set(opt.upscaleFactor, "dataset", (r, g, b))
    test_set = get_test_set(opt.upscaleFactor, "dataset", (r, g, b))

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.nbThreads, batch_size=opt.batchSize,
                                      shuffle=True)

    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.nbThreads, batch_size=opt.testBatchSize,
                                     shuffle=False)

    print('===> Building model')
    model = Net(upscale_factor=opt.upscaleFactor).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.learningRate)

    def train(epoch, logFile):
        """
        Entrainement

        :param epoch: Itération ? TODO
        :param logFile: Fichier où log
        """
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            input_pic, target_pic = batch[0].to(device), batch[1].to(device)

            # _accuracies, _val_accuracies = [],[]

            optimizer.zero_grad()
            # print(input.size())
            # print(target.size())
            loss = criterion(model(input_pic), target_pic)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(
                "===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()),
                file=logFile)

            # _accuracies.append(loss.detach())

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)),
              file=logFile)

    def test(logFile):
        """
        Teste le CNN

        :param logFile: Fichier où log

        :return: PSNR moyen
        """

        avg_psnr = 0
        with torch.no_grad():
            for batch in testing_data_loader:
                input_pic, target_pic = batch[0].to(device), batch[1].to(device)

                prediction = model(input_pic)
                mse = criterion(prediction, target_pic)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)), file=logFile)

        return avg_psnr / len(testing_data_loader)

    def checkpoint(epoch, save_dir, last=False):
        """
        Sauvegarde l'état du CNN dans un fichier .pth

        :param epoch: Itération ? TODO
        :param save_dir: Dossier de sauvegarde
        :param last: Si ce checkpoint est le dernier ou non
        """

        model_out_path = save_dir

        if last:
            model_out_path += "/model_epoch_last_{}.pth".format(letter)
        else:
            model_out_path += "/model_epoch_{}_{}.pth".format(epoch, letter)

        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    out_because_of_psnr = False

    for epoch in range(1, opt.nbEpochs + 1):
        logFile = open(save_dir + "/model_epoch_{}_{}.log".format(epoch, letter), 'w')

        train(epoch, logFile)
        avg_psnr = test(logFile)
        checkpoint(epoch, save_dir)

        logFile.close()

        print(round(avg_psnr, 2), "dB")

        if avg_psnr >= opt.noiseLimit:
            out_because_of_psnr = True
            break

    checkpoint(None, save_dir, True)

    logFile = open(save_dir + "/model_epoch_end_{}.log".format(letter), 'w')

    if out_because_of_psnr:
        print("PSNR atteint")
        print("PSNR atteint", file=logFile)
    else:
        print("epoch atteint")
        print("epoch atteint", file=logFile)


if __name__ == "__main__":
    # Paramètres
    parser = argparse.ArgumentParser(description="Entrainement du CNN")
    parser.add_argument('-u', '--upscaleFactor', type=int, required=True, help="Facteur de super-résolution")
    parser.add_argument('--batchSize', type=int, default=16, help="Taille du batch d'entrainement")
    parser.add_argument('--testBatchSize', type=int, default=10, help="Taille du batch de test")
    parser.add_argument('-n', '--nbEpochs', type=int, default=50,
                        help="Nombre de simulations max. 0 pour désactiver la limite")
    parser.add_argument('--learningRate', type=float, default=0.01, help="Taux d'apprentissage")
    parser.add_argument('--cpu', action='store_true', help="Utilise le CPU et non le GPU / CUDA")
    parser.add_argument('--nbThreads', type=int, default=4, help="Nombre de threads pour le data loader")
    parser.add_argument('--seed', type=int, default=0, help="Seed utilisée pour l'aléatoire")
    parser.add_argument('--noiseLimit', type=float, default=25,
                        help="Peak signal-to-noise ratio (PSNR),"
                             "détermine la précision du modèle en dB,"
                             "0 pour pas de limite."
                             "PSNR élevé = modèle précis")
    parser.add_argument('-q', '--quiet', action='store_true', help="Silencieux")

    opt = parser.parse_args()

    print(opt)

    if not opt.cpu and not torch.cuda.is_available():
        raise Exception("Erreur CUDA : pas de GPU trouvée. Utilisez --cpu")

    if opt.nbEpochs == 0:
        opt.nbEpochs = float("inf")

    if opt.noiseLimit == 0:
        opt.noiseLimit = float("inf")

    save_dir = "saved_model_u{}_bs{}_tbs{}_lr{}".format(opt.upscaleFactor, opt.batchSize, opt.testBatchSize,
                                                        opt.learningRate)

    print("dir name :", save_dir)

    if os.path.isdir(save_dir):
        override = "no"

        try:
            override = input("Le dossier existe déjà, voulez-vous le supprimer ? (y/N) >")
        except (EOFError, KeyboardInterrupt):
            pass

        if override.lower() in ("yes", "y"):
            print("Suppression")
            shutil.rmtree(save_dir)
        else:
            print("Annulation")
            quit()

    os.mkdir(save_dir)

    main(opt, save_dir, r=True)
    print("R OK")

    main(opt, save_dir, g=True)
    print("G OK")

    main(opt, save_dir, b=True)
    print("B OK")

# python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 100 --lr 0.001

# python main.py --upscale_factor 3 --nEpochs 100

# python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_30.pth --output_filename out.png
