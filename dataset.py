import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image

from random import shuffle

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

    def __init__(self, image_dir, input_transform=None, target_transform=None, limit=None):
        """
        Constructeur

        :param image_dir: Dossier d'images
        :param input_transform: Transformation à appliquer sur chaque entrée
        :param target_transform: Transformation à appliquer sur chaque cible
        :param limit: Limite du nombre de fichiers
        """

        super(DatasetFromFolder, self).__init__()

        input_dir = image_dir + "/input"
        target_dir = image_dir + "/target"

        input_filenames = [x for x in listdir(input_dir) if is_image_file(x)]
        target_filenames = [x for x in listdir(target_dir) if is_image_file(x)]

        image_filenames = []

        for name in input_filenames:
            if name in target_filenames and name not in image_filenames:
                image_filenames.append(name)

        for name in target_filenames:
            if name in input_filenames and name not in image_filenames:
                image_filenames.append(name)

        shuffle(image_filenames)

        if limit is not None:
            image_filenames = image_filenames[:limit]

        self.images = []

        for name in image_filenames:
            input_pic = load_img(join(input_dir, name))
            target_pic = load_img(join(target_dir, name))

            input_pic = input_transform(input_pic)
            target_pic = target_transform(target_pic)

            self.images.append((input_pic, target_pic))

    def __getitem__(self, index):
        """
        Récupère une image

        :param index: Index de l'image dans la liste

        :return: Image
        """

        return self.images[index]

    def __len__(self):
        """
        Taille du Dataset

        :return: Taille
        """

        return len(self.images)
