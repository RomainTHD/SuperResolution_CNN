from torchvision.transforms import Compose, CenterCrop, ToTensor

from dataset import DatasetFromFolder


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
        CenterCrop(crop_size * upscale_factor),
        # Resize(crop_size * upscale_factor),
        ToTensor(),
    ])


def get_training_set(upscale_factor, root_dir):
    """
    Récupère le set d'images pour l'entrainement

    :param upscale_factor: Facteur d'échelle
    :param root_dir: Dossier des images pour l'entrainement

    :return: Set d'images
    """

    crop_size = calculate_valid_crop_size(256, upscale_factor)

    print(crop_size)

    return DatasetFromFolder(root_dir,
                             input_transform=input_transform(crop_size),
                             target_transform=target_transform(crop_size, upscale_factor),
                             limit=200)


def get_test_set(upscale_factor, root_dir):
    """
    Récupère le set d'images pour le test

    :param upscale_factor: Facteur d'échelle
    :param root_dir: Dossier des images pour le test

    :return: Set d'images
    """

    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(root_dir,
                             input_transform=input_transform(crop_size),
                             target_transform=target_transform(crop_size, upscale_factor),
                             limit=200)
