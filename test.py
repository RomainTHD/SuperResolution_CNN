import subprocess
import os

DIR = "saved_model_u3_bs64_tbs10_lr0.001"

all_files = os.listdir(DIR)

for i in range(len(all_files)-1, -1, -1):
    if all_files[i].split('.')[-1] != 'pth':
        del all_files[i]

for file in all_files:
    subprocess.call(["python",
                     "super_resolve.py",
                     "--cuda",
                     "--input",
                     "input.jpg",
                     "--model",
                     DIR + "/" + file,
                     "--output",
                     "out/out_" + file.replace(".pth", "") + ".png"])

# python super_resolve.py --input_image input.jpg --model saved_model_u3_bs64_tbs10_lr1.0/model_epoch_1.pth --output_filename out/out_1.png

# python super_resolve.py --inputPath dataset/input/spr_alphys_r_3.png --modelPath saved_model_u4_bs16_tbs10_lr0.01/model_epoch_last.pth
