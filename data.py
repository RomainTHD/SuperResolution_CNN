from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder


def download_bsd300(dest="dataset"):
    """
    Télécharge le dataset BSD300

    :param dest: Destination

    :return: Chemin du dossier des images
    """
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    """
    Calcule la bonne taille pour rogner ? TODO

    :param crop_size:
    :param upscale_factor:

    :return:
    """
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size):
    """
    Retourne la transformation à appliquer sur l'image d'entrée

    :param crop_size: Taille

    :return: Transformation
    """

    return Compose([
        CenterCrop(crop_size),
        # Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size, upscale_factor):
    """
    Retourne la transformation à appliquer sur l'image cible

    :param crop_size: Taille
    :param upscale_factor: Facteur d'échelle

    :return: Transformation
    """

    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size * upscale_factor),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    """
    Récupère le set d'images pour l'entrainement

    :param upscale_factor: Facteur d'échelle

    :return: Set d'images
    """

    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size),
                             target_transform=target_transform(crop_size, upscale_factor))


def get_test_set(upscale_factor):
    """
    Récupère le set d'images pour le test

    :param upscale_factor: Facteur d'échelle

    :return: Set d'images
    """

    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size),
                             target_transform=target_transform(crop_size, upscale_factor))
