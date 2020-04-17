import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
    """
    Si un fichier est une image ou nom

    :param filename: Nom du fichier

    :return: Image ou non
    """

    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    """
    Charge une image en mode YCbCr

    :param filepath: Chemin de l'image à charger

    :return: La composante Y de l'image
    """
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    """Dataset contenant les images"""

    def __init__(self, image_dir, input_transform=None, target_transform=None):
        """
        Constructeur

        :param image_dir: Dossier d'images
        :param input_transform: Transformation à appliquer sur chaque entrée
        :param target_transform: Transformation à appliquer sur chaque cible
        """

        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Récupère une image

        :param index: Index de l'image dans la liste

        :return: Image
        """

        input_pic = load_img(self.image_filenames[index])
        target_pic = input_pic.copy()
        if self.input_transform:
            input_pic = self.input_transform(input_pic)
        if self.target_transform:
            target_pic = self.target_transform(target_pic)

        return input_pic, target_pic

    def __len__(self):
        """
        Taille du Dataset

        :return: Taille
        """

        return len(self.image_filenames)
