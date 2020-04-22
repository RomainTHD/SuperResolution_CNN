# Super-résolution via CNN

### Entrainement
```
Entrainement: train.py

Arguments:
  -h, --help                        Message d'aide
  --upscaleFactor     ! REQUIS      Facteur de super résolution
  --batchSize           16          Taille du batch d'entrainement
  --testBatchSize       10          Taille du batch de test
  --nbEpochs            50          Nombre de simulations max. 0 pour désactiver la limite
  --learningRate        0.01        Taux d'apprentissage
  --cuda                            Utilise la puissance GPU via Cuda (forcé à True pour l'instant)
  --nbThreads           4           Nombre de Threads pour le chargement de la data
  --seed                123         Seed à utiliser pour l'aléatoire
  --noiseLimit          25          PSNR, Précision en dB, haut = précis, 0 = pas de limite
```

Générera des fichiers model_epoch_`n`.pth correspondant au modèle à l'époque `n`,
dans un dossier saved_model_u`2`_bs`64`_tbs`10`_lr`0.01` par exemple, où `2` correspond au facteur d'échelle, `64` à la taille du batch d'entrainement, `10` la taille du batch de test et `0.01` au taux d'apprentissage.

[//]: # (This example trains a super-resolution network on the , using crops from the 200 training images, and evaluating on crops of the 100 test images. A snapshot of the model after every epoch with filename model_epoch_<epoch_number>.pth)

#### Exemple :
`python train.py --upscaleFactor 4 --nbEpochs 20`

[//]: # (`python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001`)

### Super-résolution
```
Super-résolution: super_resolve.py

Arguments:
  -h, --help                        Message d'aide
  --inputPath         ! REQUIS      Image d'entrée
  --modelPath         ! REQUIS      Modèle à utiliser
  --outputPath          out.png     Image de sortie
  --cuda                            Utilise la puissance GPU via Cuda (forcé à True pour l'instant)
```

#### Exemple :
`python super_resolve.py --inputPath dataset/input/spr_alphys_r_3.png --modelPath saved_model_u4_bs16_tbs10_lr0.01/model_epoch_last.pth`

[//]: # (`python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_500.pth --output_filename out.png`)

<hr>

### Licence

CC BY-NC-SA 4.0 <br>
Creative Commons Attribution - Pas d’Utilisation Commerciale - Partage dans les Mêmes Conditions 4.0 International

<hr>

### Crédits

Inspiré de [<u>cet article</u>](https://arxiv.org/abs/1609.05158)
<br>
Code de base disponible [<u>ici</u>](https://github.com/pytorch/examples/tree/master/super_resolution)
<br>
Dataset cible construit à partir de l'algorithme [<u>GTVimageVect</u>](https://github.com/kerautret/GTVimageVect)
<br>
Sujet disponible [<u>ici</u>](https://www.lama.univ-savoie.fr/mediawiki/index.php/VISI401_CMI_:_bibliographie_scientifique#Algorithmes_de_super-r.C3.A9solution_par_apprentissage)

[//]: # ("Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.)

Romain THEODET, CMI INFO L2 USMB, 2019/2020
