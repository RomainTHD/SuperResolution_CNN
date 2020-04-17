"""Modèle du CNN"""

import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    """Réseau CNN"""

    def __init__(self, upscale_factor):
        """Initialisation

        :param upscale_factor: Facteur d'échelle
        """

        super(Net, self).__init__()

        # Fonction pour le forward
        self.relu = nn.ReLU()

        # TODO arguments suivant ?
        # On part de 1 bloc qu'on divise en 64 masques de blocs 5*5
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))

        # on refait 64 masques de blocs 3*3
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Puis 32 masques de blocs 3*3
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))

        # Puis (upscale_factor ** 2) masques de blocs 3*3
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))

        # TODO
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        """Avance au cycle suivant

        :param x: TODO

        :return: TODO
        """

        # print("forward 1", x.size())
        x = self.relu(self.conv1(x))
        # print("forward 2", x.size())
        x = self.relu(self.conv2(x))
        # print("forward 3", x.size())
        x = self.relu(self.conv3(x))
        # print("forward 4", x.size())
        x = self.pixel_shuffle(self.conv4(x))
        # print("forward 5", x.size())

        return x

    def _initialize_weights(self):
        """Initialise les poids"""

        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
