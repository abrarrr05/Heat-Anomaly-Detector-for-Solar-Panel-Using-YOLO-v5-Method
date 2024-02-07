!pip install -q gradio==3.50.2

import gradio as gr
import os

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

rf = Roboflow(api_key="nF9pI0M2rb5a7i*****")
project = rf.workspace("orbit-future-academy-*****").project("*****-8lyfl")
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

from detect import run

# Download Weight (Pilih Folder Best.pt nya !)
gdown.download_folder("https://drive.google.com/drive/folders/1DfeuNCCh-CMg6bCCs7xnvqVMXgeN4LYQ?usp=sharing")

"""## Inference Satu Gambar"""

def yolov5_inference(image,iou_threshold,confidence_threshold):
    run(weights='/content/yolov5/Best Weight/best.pt',
        imgsz=(640,640),
        iou_thres=iou_threshold,
        conf_thres=confidence_threshold,
        exist_ok=True,
        line_thickness=2,
        source=image)

    result_image = gr.Image(glob.glob('/content/yolov5/runs/detect/exp/*.png')[0])
    return result_image

with gr.Blocks() as app:
    gr.Markdown(
    """
    <img src="https://lldikti10.id/public/img/informasi/berita/MASTER.png" alt="Logo Kampus Merdeka" style="height: 50px;"/>


    # Aplikasiku !
    Silahkan Pilih atau Unggah Gambar dan Kemudian Pilih "Submit". Terima Kasih :)

    """),

    gr.Interface(
    fn=yolov5_inference,
    inputs=[gr.Image(type="filepath"),gr.Slider(0, 1,step=0.1,value=0.5),gr.Slider(0, 1,step=0.1,value=0.5)],
    outputs="image",
    allow_flagging="never",
    examples=[
        [os.path.join(os.path.abspath(''), "/content/yolov5/HADSOL-1/train/images/1_jpg.rf.94923f714f745d3a959f78cebe772d54.jpg"),0.5,0.2],
        [os.path.join(os.path.abspath(''), "/content/yolov5/HADSOL-1/train/images/2_jpg.rf.0625a0212484af37aa8f131542305218.jpg"),0.5,0.2],
        [os.path.join(os.path.abspath(''), "/content/yolov5/HADSOL-1/train/images/396792-fig-003a_jpg.rf.0aa07190cb64d16a4dad457ce7329a26.jpg"),0.5,0.2],
        [os.path.join(os.path.abspath(''), "/content/yolov5/HADSOL-1/train/images/3_jpg.rf.00e2b9618463604ed42d5d7fdebe7ec5.jpg"),0.5,0.1]
    ],
)

app.launch(debug=False, share=True)