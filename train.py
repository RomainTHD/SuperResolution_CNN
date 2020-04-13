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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--upscaleFactor', type=int, required=True, help="super resolution upscale factor")
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
    parser.add_argument('--nbEpochs', type=int, default=100, help='number of epochs to train for, 0 pour pas de limite')
    parser.add_argument('--learningRate', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--nbThreads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--noiseLimit', type=float, default=25, help="Peak signal-to-noise ratio (PSNR), précision en dB, petit = affreux, 0 pour pas de limite")
    opt = parser.parse_args()

    print(opt)

    opt.cuda = True

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if opt.nbEpochs == 0:
        opt.nbEpochs = float("inf")

    if opt.noiseLimit == 0:
        opt.noiseLimit = float("inf")

    save_dir = "saved_model_u{}_bs{}_tbs{}_lr{}".format(opt.upscaleFactor, opt.batchSize, opt.testBatchSize, opt.learningRate)

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

    f = open(save_dir + "/args.cfg", 'w')
    print(opt, file=f)
    f.close()

    torch.manual_seed(opt.seed)

    device = torch.device("cuda" if opt.cuda else "cpu")

    print('===> Loading datasets')
    train_set = get_training_set(opt.upscaleFactor)
    test_set = get_test_set(opt.upscaleFactor)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.nbThreads, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.nbThreads, batch_size=opt.testBatchSize, shuffle=False)

    print('===> Building model')
    model = Net(upscale_factor=opt.upscaleFactor).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.learningRate)


    def train(epoch, logFile):
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            input, target = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            loss = criterion(model(input), target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()), file=logFile)

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)), file=logFile)


    def test(logFile):
        avg_psnr = 0
        with torch.no_grad():
            for batch in testing_data_loader:
                input, target = batch[0].to(device), batch[1].to(device)

                prediction = model(input)
                mse = criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)), file=logFile)

        return avg_psnr / len(testing_data_loader)

    def checkpoint(epoch, save_dir):
        model_out_path = save_dir + "/model_epoch_{}.pth".format(epoch)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    out_because_of_psnr = False

    for epoch in range(1, opt.nbEpochs + 1):
        logFile = open(save_dir + "/model_epoch_{}.log".format(epoch), 'w')

        train(epoch, logFile)
        avg_psnr = test(logFile)
        checkpoint(epoch, save_dir)

        logFile.close()

        print(round(avg_psnr, 2), "dB")

        if avg_psnr >= opt.noiseLimit:
            out_because_of_psnr = True
            break

    logFile = open(save_dir + "/model_epoch_end.log", 'w')

    if out_because_of_psnr:
        print("PSNR atteint")
        print("PSNR atteint", file=logFile)
    else:
        print("epoch atteint")
        print("epoch atteint", file=logFile)

if __name__ == "__main__":
    main()

# python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 100 --lr 0.001

# python main.py --upscale_factor 3 --nEpochs 100

# python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_30.pth --output_filename out.png
