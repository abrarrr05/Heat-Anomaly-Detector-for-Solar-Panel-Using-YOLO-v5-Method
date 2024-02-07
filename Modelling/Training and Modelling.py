
# Commented out IPython magic to ensure Python compatibility.
# clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5  # clone repo
# %cd yolov5

# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch

from IPython.display import Image, clear_output  # to display images
from utils.downloads import attempt_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

!pip install -q roboflow

"""# Download Dataset dari Roboflow"""

# Commented out IPython magic to ensure Python compatibility.
# follow the link below to get your download code from from Roboflow
# %cd /content/yolov5

from roboflow import Roboflow

rf = Roboflow(api_key="nF9pI0M2rb5a7iu*****")
project = rf.workspace("orbit-future-academy-*****").project("******-8lyfl")
dataset = project.version(1).download("yolov5")

"""# Define Model Configuration and Architecture

"""

# Commented out IPython magic to ensure Python compatibility.
# this is the YAML file Roboflow wrote for us that we're loading into this notebook with our data
# %cat {dataset.location}/data.yaml

# define number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

# Commented out IPython magic to ensure Python compatibility.
#this is the model configuration we will use for our tutorial
# %cat /content/yolov5/models/yolov5s.yaml

"""# Inference"""

import gdown
import glob
from IPython.display import Image, display
import shutil
from google.colab import files

# Download Weight (Pilih Folder Best.pt nya !)
gdown.download_folder("https://drive.google.com/drive/folders/1DfeuNCCh-CMg6bCCs7xnvqVMXgeN4LYQ?usp=sharing")

"""## Inference Satu Gambar"""

!python detect.py --weights '/content/yolov5/Best Weight/best.pt' --img 640 --iou-thres 0.5 --conf-thres 0.4 --exist-ok --line-thickness 2 --source '/content/yolov5/HADSOL-1/train/images/1_jpg.rf.94923f714f745d3a959f78cebe772d54.jpg'

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg')[0:1]:
    display(Image(filename=imageName))
    print("\n")

# zip hasil inference dan download
shutil.make_archive('/content/download_inference_single_file', 'zip', '/content/yolov5/runs/detect/exp/')
files.download('/content/download_inference_single_file.zip')

# Hapus Semua File Hasil Deteksi Jika Tidak Dibutuhkan
!rm /content/yolov5/runs/detect/exp/*

"""## Inference Satu Folder"""

!python detect.py --weights '/content/yolov5/Best Weight/best.pt' --img 640 --iou-thres 0.5 --conf 0.4 --exist-ok --line-thickness 2 --source '/content/yolov5/HADSOL-1/test/images/'

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg')[0:10]: # Kita tampilkan 10 Gambar Test Saja
    display(Image(filename=imageName))
    print("\n")

# zip hasil inference dan download
shutil.make_archive('/content/download_inference_folder', 'zip', '/content/yolov5/runs/detect/exp/')
files.download('/content/download_inference_folder.zip')

# Hapus Semua File Hasil Deteksi Jika Tidak Dibutuhkan
!rm /content/yolov5/runs/detect/exp/*

"""# Evaluasi"""

!cp {dataset.location}/data.yaml {dataset.location}/val.yaml

"""## Train"""

!sed -i 's@HADSOL-1/valid/images@HADSOL-1/train/images@g' {dataset.location}/val.yaml

!python val.py --weights '/content/yolov5/Best Weight/best.pt' --img 640 --exist-ok --name 'train' --data {dataset.location}/val.yaml

# zip hasil evaluasi dan download
shutil.make_archive('/content/download_eval_train', 'zip', '/content/yolov5/runs/val/train')
files.download('/content/download_eval_train.zip')

"""## Validation"""

!sed -i 's@HADSOL-1/train/images@HADSOL-1/valid/images@g' {dataset.location}/val.yaml

!python val.py --weights '/content/yolov5/Best Weight/best.pt' --img 640 --exist-ok --name 'validation' --data {dataset.location}/val.yaml

# zip hasil evaluasi dan download
shutil.make_archive('/content/download_eval_validation', 'zip', '/content/yolov5/runs/val/validation')
files.download('/content/download_eval_validation.zip')

"""## Test"""

!sed -i 's@HADSOL-1/valid/images@HADSOL-1/test/images@g' {dataset.location}/val.yaml

!python val.py --weights '/content/yolov5/Best Weight/best.pt' --img 640 --exist-ok --name 'test' --data {dataset.location}/val.yaml

# zip hasil evaluasi dan download
shutil.make_archive('/content/download_eval_test', 'zip', '/content/yolov5/runs/val/test')
files.download('/content/download_eval_test.zip')