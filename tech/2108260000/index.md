---
layout: post
type: tech
date: 2021-08-26 15:21
title: 경복궁 건숙문 객체 검출
subtitle: 주석화 - polygon방식
---

# **detectron2 모델을 사용한 건숙문 객체 모델 학습**


*   👈 드라이브 연동(mount)
*   드라이브 연동 후 현재 .ipynb 파일이 있는 경로로 이동 👇




```python
%cd /content/drive/MyDrive/한양도성/detectron2/detectron2_gsm
```

    



*   torch 설치 👇




```python
!pip install -U torch torchvision
!pip install git+http://github.com/facebookresearch/fvcore.git
import torch, torchvision
torch.__version__
```

    Requirement already satisfied: torch in /usr/local/lib/python3.7/dist-packages (1.9.0+cu102)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.7/dist-packages (0.10.0+cu102)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch) (3.7.4.3)
    Requirement already satisfied: pillow>=5.3.0 in /usr/local/lib/python3.7/dist-packages (from torchvision) (7.1.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from torchvision) (1.19.5)
    Collecting git+http://github.com/facebookresearch/fvcore.git
      Cloning http://github.com/facebookresearch/fvcore.git to /tmp/pip-req-build-qu9_gdmc
      Running command git clone -q http://github.com/facebookresearch/fvcore.git /tmp/pip-req-build-qu9_gdmc
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from fvcore==0.1.5) (1.19.5)
    Collecting yacs>=0.1.6
      Downloading yacs-0.1.8-py3-none-any.whl (14 kB)
    Collecting pyyaml>=5.1
      Downloading PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)
    [K     |████████████████████████████████| 636 kB 9.0 MB/s 
    [?25hRequirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from fvcore==0.1.5) (4.62.0)
    Requirement already satisfied: termcolor>=1.1 in /usr/local/lib/python3.7/dist-packages (from fvcore==0.1.5) (1.1.0)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from fvcore==0.1.5) (7.1.2)
    Requirement already satisfied: tabulate in /usr/local/lib/python3.7/dist-packages (from fvcore==0.1.5) (0.8.9)
    Collecting iopath>=0.1.7
      Downloading iopath-0.1.9-py3-none-any.whl (27 kB)
    Collecting portalocker
      Downloading portalocker-2.3.0-py2.py3-none-any.whl (15 kB)
    Building wheels for collected packages: fvcore
      Building wheel for fvcore (setup.py) ... [?25l[?25hdone
      Created wheel for fvcore: filename=fvcore-0.1.5-py3-none-any.whl size=64549 sha256=051cee7766d1f2360d2637a810135d42f08e7034f4d347d1700472e9d647d31f
      Stored in directory: /tmp/pip-ephem-wheel-cache-2gam5n50/wheels/84/a2/39/c52aa1b0182707ed2948f91bbc35bc63abd4f3156915f6b280
    Successfully built fvcore
    Installing collected packages: pyyaml, portalocker, yacs, iopath, fvcore
      Attempting uninstall: pyyaml
        Found existing installation: PyYAML 3.13
        Uninstalling PyYAML-3.13:
          Successfully uninstalled PyYAML-3.13
    Successfully installed fvcore-0.1.5 iopath-0.1.9 portalocker-2.3.0 pyyaml-5.4.1 yacs-0.1.8
    




    '1.9.0+cu102'



* 런타임 다시 시작 

* detectron2 모델을 사용하기 위한 git저장소 clone 👇


```python
%cd /content/drive/MyDrive/한양도성/detectron2/detectron2_gsm
!git clone https://github.com/facebookresearch/detectron2 detectron2_repo
!pip install -e detectron2_repo
```

    /content/drive/MyDrive/한양도성/detectron2/detectron2_gsm
    fatal: destination path 'detectron2_repo' already exists and is not an empty directory.
    Obtaining file:///content/drive/MyDrive/%E1%84%92%E1%85%A1%E1%86%AB%E1%84%8B%E1%85%A3%E1%86%BC%E1%84%83%E1%85%A9%E1%84%89%E1%85%A5%E1%86%BC/detectron2/detectron2_gsm/detectron2_repo
    Requirement already satisfied: Pillow>=7.1 in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (7.1.2)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (3.2.2)
    Requirement already satisfied: pycocotools>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (2.0.2)
    Requirement already satisfied: termcolor>=1.1 in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (1.1.0)
    Requirement already satisfied: yacs>=0.1.6 in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (0.1.8)
    Requirement already satisfied: tabulate in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (0.8.9)
    Requirement already satisfied: cloudpickle in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (1.3.0)
    Requirement already satisfied: tqdm>4.29.0 in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (4.62.0)
    Requirement already satisfied: tensorboard in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (2.6.0)
    Requirement already satisfied: fvcore<0.1.6,>=0.1.5 in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (0.1.5)
    Requirement already satisfied: iopath<0.1.10,>=0.1.7 in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (0.1.9)
    Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (0.16.0)
    Requirement already satisfied: pydot in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (1.3.0)
    Requirement already satisfied: omegaconf>=2.1 in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (2.1.0)
    Requirement already satisfied: hydra-core>=1.1 in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (1.1.0)
    Requirement already satisfied: black==21.4b2 in /usr/local/lib/python3.7/dist-packages (from detectron2==0.5) (21.4b2)
    Requirement already satisfied: typing-extensions>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from black==21.4b2->detectron2==0.5) (3.7.4.3)
    Requirement already satisfied: toml>=0.10.1 in /usr/local/lib/python3.7/dist-packages (from black==21.4b2->detectron2==0.5) (0.10.2)
    Requirement already satisfied: mypy-extensions>=0.4.3 in /usr/local/lib/python3.7/dist-packages (from black==21.4b2->detectron2==0.5) (0.4.3)
    Requirement already satisfied: appdirs in /usr/local/lib/python3.7/dist-packages (from black==21.4b2->detectron2==0.5) (1.4.4)
    Requirement already satisfied: typed-ast>=1.4.2 in /usr/local/lib/python3.7/dist-packages (from black==21.4b2->detectron2==0.5) (1.4.3)
    Requirement already satisfied: regex>=2020.1.8 in /usr/local/lib/python3.7/dist-packages (from black==21.4b2->detectron2==0.5) (2021.8.3)
    Requirement already satisfied: click>=7.1.2 in /usr/local/lib/python3.7/dist-packages (from black==21.4b2->detectron2==0.5) (7.1.2)
    Requirement already satisfied: pathspec<1,>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from black==21.4b2->detectron2==0.5) (0.9.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from fvcore<0.1.6,>=0.1.5->detectron2==0.5) (1.19.5)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.7/dist-packages (from fvcore<0.1.6,>=0.1.5->detectron2==0.5) (5.4.1)
    Requirement already satisfied: importlib-resources in /usr/local/lib/python3.7/dist-packages (from hydra-core>=1.1->detectron2==0.5) (5.2.2)
    Requirement already satisfied: antlr4-python3-runtime==4.8 in /usr/local/lib/python3.7/dist-packages (from hydra-core>=1.1->detectron2==0.5) (4.8)
    Requirement already satisfied: portalocker in /usr/local/lib/python3.7/dist-packages (from iopath<0.1.10,>=0.1.7->detectron2==0.5) (2.3.0)
    Requirement already satisfied: cython>=0.27.3 in /usr/local/lib/python3.7/dist-packages (from pycocotools>=2.0.2->detectron2==0.5) (0.29.24)
    Requirement already satisfied: setuptools>=18.0 in /usr/local/lib/python3.7/dist-packages (from pycocotools>=2.0.2->detectron2==0.5) (57.4.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->detectron2==0.5) (2.8.2)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->detectron2==0.5) (2.4.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->detectron2==0.5) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->detectron2==0.5) (0.10.0)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from cycler>=0.10->matplotlib->detectron2==0.5) (1.15.0)
    Requirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.7/dist-packages (from importlib-resources->hydra-core>=1.1->detectron2==0.5) (3.5.0)
    Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (0.37.0)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (0.6.1)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (0.4.5)
    Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (0.12.0)
    Requirement already satisfied: google-auth<2,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (1.34.0)
    Requirement already satisfied: grpcio>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (1.39.0)
    Requirement already satisfied: protobuf>=3.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (3.17.3)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (1.8.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (3.3.4)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (2.23.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard->detectron2==0.5) (1.0.1)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard->detectron2==0.5) (4.2.2)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard->detectron2==0.5) (4.7.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard->detectron2==0.5) (0.2.8)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard->detectron2==0.5) (1.3.0)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard->detectron2==0.5) (4.6.4)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard->detectron2==0.5) (0.4.8)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard->detectron2==0.5) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard->detectron2==0.5) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard->detectron2==0.5) (2021.5.30)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard->detectron2==0.5) (1.24.3)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard->detectron2==0.5) (3.1.1)
    Installing collected packages: detectron2
      Attempting uninstall: detectron2
        Found existing installation: detectron2 0.5
        Can't uninstall 'detectron2'. No files were found to uninstall.
      Running setup.py develop for detectron2
    Successfully installed detectron2-0.5
    



*   필요한 라이브러리 import 하기 👇




```python
%cd /content/drive/MyDrive/한양도성/detectron2/detectron2_gsm

# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
```

    /content/drive/MyDrive/한양도성/detectron2/detectron2_gsm
    

* coco 데이터셋에 gsm(건숙문) 객체 instance를 추가 👇


```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("gsm", {}, "./data/trainval.json", "./data/images")
person_metadata = MetadataCatalog.get("gsm")
dataset_dicts = DatasetCatalog.get("gsm")
```

    [5m[31mWARNING[0m [32m[08/19 07:10:12 d2.data.datasets.coco]: [0m
    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
    
    [32m[08/19 07:10:12 d2.data.datasets.coco]: [0mLoaded 55 images in COCO format from ./data/trainval.json
    

* 55개의 훈련이미지 중 1개 랜덤 확인(주석화가 잘 되었는지) 👇


```python
import random

for d in random.sample(dataset_dicts, 1):

    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=person_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])
```


    
![png](./img/output_12_0.png)
    


* 훈련 준비 및 훈련하기<br>
iteration : 300


```python
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file("/content/drive/MyDrive/한양도성/detectron2/detectron2_gsm/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("gsm",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (gsm)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

    [32m[08/19 07:11:25 d2.engine.defaults]: [0mModel:
    GeneralizedRCNN(
      (backbone): FPN(
        (fpn_lateral2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (fpn_output2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (fpn_lateral3): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (fpn_output3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (fpn_lateral4): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        (fpn_output4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (fpn_lateral5): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        (fpn_output5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (top_block): LastLevelMaxPool()
        (bottom_up): ResNet(
          (stem): BasicStem(
            (conv1): Conv2d(
              3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
              (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
            )
          )
          (res2): Sequential(
            (0): BottleneckBlock(
              (shortcut): Conv2d(
                64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv1): Conv2d(
                64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv2): Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv3): Conv2d(
                64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
            )
            (1): BottleneckBlock(
              (conv1): Conv2d(
                256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv2): Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv3): Conv2d(
                64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
            )
            (2): BottleneckBlock(
              (conv1): Conv2d(
                256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv2): Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv3): Conv2d(
                64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
            )
          )
          (res3): Sequential(
            (0): BottleneckBlock(
              (shortcut): Conv2d(
                256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv1): Conv2d(
                256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv2): Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv3): Conv2d(
                128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
            )
            (1): BottleneckBlock(
              (conv1): Conv2d(
                512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv2): Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv3): Conv2d(
                128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
            )
            (2): BottleneckBlock(
              (conv1): Conv2d(
                512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv2): Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv3): Conv2d(
                128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
            )
            (3): BottleneckBlock(
              (conv1): Conv2d(
                512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv2): Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv3): Conv2d(
                128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
            )
          )
          (res4): Sequential(
            (0): BottleneckBlock(
              (shortcut): Conv2d(
                512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
              (conv1): Conv2d(
                512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (1): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (2): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (3): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (4): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (5): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
          )
          (res5): Sequential(
            (0): BottleneckBlock(
              (shortcut): Conv2d(
                1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
              )
              (conv1): Conv2d(
                1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv2): Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv3): Conv2d(
                512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
              )
            )
            (1): BottleneckBlock(
              (conv1): Conv2d(
                2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv2): Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv3): Conv2d(
                512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
              )
            )
            (2): BottleneckBlock(
              (conv1): Conv2d(
                2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv2): Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv3): Conv2d(
                512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
              )
            )
          )
        )
      )
      (proposal_generator): RPN(
        (rpn_head): StandardRPNHead(
          (conv): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            (activation): ReLU()
          )
          (objectness_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
          (anchor_deltas): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
        )
        (anchor_generator): DefaultAnchorGenerator(
          (cell_anchors): BufferList()
        )
      )
      (roi_heads): StandardROIHeads(
        (box_pooler): ROIPooler(
          (level_poolers): ModuleList(
            (0): ROIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=0, aligned=True)
            (1): ROIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=0, aligned=True)
            (2): ROIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
            (3): ROIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=0, aligned=True)
          )
        )
        (box_head): FastRCNNConvFCHead(
          (flatten): Flatten(start_dim=1, end_dim=-1)
          (fc1): Linear(in_features=12544, out_features=1024, bias=True)
          (fc_relu1): ReLU()
          (fc2): Linear(in_features=1024, out_features=1024, bias=True)
          (fc_relu2): ReLU()
        )
        (box_predictor): FastRCNNOutputLayers(
          (cls_score): Linear(in_features=1024, out_features=2, bias=True)
          (bbox_pred): Linear(in_features=1024, out_features=4, bias=True)
        )
        (mask_pooler): ROIPooler(
          (level_poolers): ModuleList(
            (0): ROIAlign(output_size=(14, 14), spatial_scale=0.25, sampling_ratio=0, aligned=True)
            (1): ROIAlign(output_size=(14, 14), spatial_scale=0.125, sampling_ratio=0, aligned=True)
            (2): ROIAlign(output_size=(14, 14), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
            (3): ROIAlign(output_size=(14, 14), spatial_scale=0.03125, sampling_ratio=0, aligned=True)
          )
        )
        (mask_head): MaskRCNNConvUpsampleHead(
          (mask_fcn1): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            (activation): ReLU()
          )
          (mask_fcn2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            (activation): ReLU()
          )
          (mask_fcn3): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            (activation): ReLU()
          )
          (mask_fcn4): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            (activation): ReLU()
          )
          (deconv): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
          (deconv_relu): ReLU()
          (predictor): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    [5m[31mWARNING[0m [32m[08/19 07:11:26 d2.data.datasets.coco]: [0m
    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
    
    [32m[08/19 07:11:26 d2.data.datasets.coco]: [0mLoaded 55 images in COCO format from ./data/trainval.json
    [32m[08/19 07:11:26 d2.data.build]: [0mRemoved 0 images with no usable annotations. 55 images left.
    [32m[08/19 07:11:26 d2.data.build]: [0mDistribution of instances among all 1 categories:
    [36m|  category  | #instances   |
    |:----------:|:-------------|
    |    gsm     | 55           |
    |            |              |[0m
    [32m[08/19 07:11:26 d2.data.dataset_mapper]: [0m[DatasetMapper] Augmentations used in training: [ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'), RandomFlip()]
    [32m[08/19 07:11:26 d2.data.build]: [0mUsing training sampler TrainingSampler
    [32m[08/19 07:11:26 d2.data.common]: [0mSerializing 55 elements to byte tensors and concatenating them all ...
    [32m[08/19 07:11:26 d2.data.common]: [0mSerialized dataset takes 0.08 MiB
    [5m[31mWARNING[0m [32m[08/19 07:11:26 d2.solver.build]: [0mSOLVER.STEPS contains values larger than SOLVER.MAX_ITER. These values will be ignored.
    

    model_final_f10217.pkl: 178MB [00:02, 64.7MB/s]                           
    Skip loading parameter 'roi_heads.box_predictor.cls_score.weight' to the model due to incompatible shapes: (81, 1024) in the checkpoint but (2, 1024) in the model! You might want to double check if this is expected.
    Skip loading parameter 'roi_heads.box_predictor.cls_score.bias' to the model due to incompatible shapes: (81,) in the checkpoint but (2,) in the model! You might want to double check if this is expected.
    Skip loading parameter 'roi_heads.box_predictor.bbox_pred.weight' to the model due to incompatible shapes: (320, 1024) in the checkpoint but (4, 1024) in the model! You might want to double check if this is expected.
    Skip loading parameter 'roi_heads.box_predictor.bbox_pred.bias' to the model due to incompatible shapes: (320,) in the checkpoint but (4,) in the model! You might want to double check if this is expected.
    Skip loading parameter 'roi_heads.mask_head.predictor.weight' to the model due to incompatible shapes: (80, 256, 1, 1) in the checkpoint but (1, 256, 1, 1) in the model! You might want to double check if this is expected.
    Skip loading parameter 'roi_heads.mask_head.predictor.bias' to the model due to incompatible shapes: (80,) in the checkpoint but (1,) in the model! You might want to double check if this is expected.
    Some model parameters or buffers are not found in the checkpoint:
    [34mroi_heads.box_predictor.bbox_pred.{bias, weight}[0m
    [34mroi_heads.box_predictor.cls_score.{bias, weight}[0m
    [34mroi_heads.mask_head.predictor.{bias, weight}[0m
    

    [32m[08/19 07:11:30 d2.engine.train_loop]: [0mStarting training from iteration 0
    

    /usr/local/lib/python3.7/dist-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  /pytorch/aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
    /usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    

    [32m[08/19 07:12:44 d2.utils.events]: [0m eta: 0:17:09  iter: 19  total_loss: 1.261  loss_cls: 0.3186  loss_box_reg: 0.3577  loss_mask: 0.6264  loss_rpn_cls: 0.004967  loss_rpn_loc: 0.006363  time: 3.6087  data_time: 3.1681  lr: 0.0012854  max_mem: 2545M
    [32m[08/19 07:13:50 d2.utils.events]: [0m eta: 0:15:07  iter: 39  total_loss: 0.6021  loss_cls: 0.07519  loss_box_reg: 0.376  loss_mask: 0.1635  loss_rpn_cls: 0.0004425  loss_rpn_loc: 0.004831  time: 3.4435  data_time: 2.7737  lr: 0.0026174  max_mem: 2545M
    [32m[08/19 07:14:51 d2.utils.events]: [0m eta: 0:12:46  iter: 59  total_loss: 0.3032  loss_cls: 0.04368  loss_box_reg: 0.1227  loss_mask: 0.1229  loss_rpn_cls: 0.0001886  loss_rpn_loc: 0.00353  time: 3.3224  data_time: 2.5453  lr: 0.0039494  max_mem: 2545M
    [32m[08/19 07:15:54 d2.utils.events]: [0m eta: 0:11:37  iter: 79  total_loss: 0.2504  loss_cls: 0.02241  loss_box_reg: 0.09888  loss_mask: 0.1108  loss_rpn_cls: 0.0005882  loss_rpn_loc: 0.003385  time: 3.2727  data_time: 2.5894  lr: 0.0052814  max_mem: 2545M
    [32m[08/19 07:16:56 d2.utils.events]: [0m eta: 0:10:31  iter: 99  total_loss: 0.238  loss_cls: 0.02274  loss_box_reg: 0.1131  loss_mask: 0.0924  loss_rpn_cls: 0.0003196  loss_rpn_loc: 0.002991  time: 3.2423  data_time: 2.5862  lr: 0.0066134  max_mem: 2545M
    [32m[08/19 07:17:58 d2.utils.events]: [0m eta: 0:09:25  iter: 119  total_loss: 0.2521  loss_cls: 0.02735  loss_box_reg: 0.1136  loss_mask: 0.1088  loss_rpn_cls: 0.0003729  loss_rpn_loc: 0.002281  time: 3.2149  data_time: 2.5429  lr: 0.0079454  max_mem: 2545M
    [32m[08/19 07:19:01 d2.utils.events]: [0m eta: 0:08:23  iter: 139  total_loss: 0.232  loss_cls: 0.02028  loss_box_reg: 0.1121  loss_mask: 0.09035  loss_rpn_cls: 0.0002983  loss_rpn_loc: 0.003012  time: 3.2028  data_time: 2.5961  lr: 0.0092774  max_mem: 2545M
    [32m[08/19 07:20:04 d2.utils.events]: [0m eta: 0:07:20  iter: 159  total_loss: 0.2147  loss_cls: 0.01992  loss_box_reg: 0.09812  loss_mask: 0.0875  loss_rpn_cls: 0.0003793  loss_rpn_loc: 0.002713  time: 3.1951  data_time: 2.5860  lr: 0.010609  max_mem: 2545M
    [32m[08/19 07:21:06 d2.utils.events]: [0m eta: 0:06:17  iter: 179  total_loss: 0.2268  loss_cls: 0.02461  loss_box_reg: 0.08976  loss_mask: 0.1104  loss_rpn_cls: 0.0006289  loss_rpn_loc: 0.002692  time: 3.1874  data_time: 2.6019  lr: 0.011941  max_mem: 2545M
    [32m[08/19 07:22:10 d2.utils.events]: [0m eta: 0:05:14  iter: 199  total_loss: 0.2229  loss_cls: 0.01733  loss_box_reg: 0.1028  loss_mask: 0.08705  loss_rpn_cls: 0.0004452  loss_rpn_loc: 0.002777  time: 3.1858  data_time: 2.6162  lr: 0.013273  max_mem: 2545M
    [32m[08/19 07:23:13 d2.utils.events]: [0m eta: 0:04:12  iter: 219  total_loss: 0.2288  loss_cls: 0.0182  loss_box_reg: 0.1056  loss_mask: 0.09144  loss_rpn_cls: 0.0005783  loss_rpn_loc: 0.002713  time: 3.1848  data_time: 2.6293  lr: 0.014605  max_mem: 2545M
    [32m[08/19 07:24:17 d2.utils.events]: [0m eta: 0:03:09  iter: 239  total_loss: 0.2629  loss_cls: 0.02343  loss_box_reg: 0.1307  loss_mask: 0.09923  loss_rpn_cls: 0.0006588  loss_rpn_loc: 0.003805  time: 3.1855  data_time: 2.6455  lr: 0.015937  max_mem: 2545M
    [32m[08/19 07:25:20 d2.utils.events]: [0m eta: 0:02:06  iter: 259  total_loss: 0.2276  loss_cls: 0.0196  loss_box_reg: 0.1036  loss_mask: 0.09085  loss_rpn_cls: 0.001046  loss_rpn_loc: 0.003121  time: 3.1825  data_time: 2.5975  lr: 0.017269  max_mem: 2545M
    [32m[08/19 07:26:23 d2.utils.events]: [0m eta: 0:01:03  iter: 279  total_loss: 0.1896  loss_cls: 0.01755  loss_box_reg: 0.08669  loss_mask: 0.08638  loss_rpn_cls: 0.0007518  loss_rpn_loc: 0.002928  time: 3.1792  data_time: 2.5827  lr: 0.018601  max_mem: 2545M
    [32m[08/19 07:27:27 d2.utils.events]: [0m eta: 0:00:00  iter: 299  total_loss: 0.191  loss_cls: 0.01448  loss_box_reg: 0.09248  loss_mask: 0.07239  loss_rpn_cls: 0.0007743  loss_rpn_loc: 0.002967  time: 3.1766  data_time: 2.5774  lr: 0.019933  max_mem: 2545M
    [32m[08/19 07:27:27 d2.engine.hooks]: [0mOverall training speed: 298 iterations in 0:15:46 (3.1766 s / it)
    [32m[08/19 07:27:27 d2.engine.hooks]: [0mTotal training time: 0:15:47 (0:00:01 on hooks)
    


```python
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file("/content/drive/MyDrive/한양도성/detectron2/detectron2_gsm/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("gsm",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (gsm)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/한양도성/detectron2/detectron2_gsm/output/model_final.pth" # 본인의 model이저장된 경로로 수정해줍니다.
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("gsm", )
predictor = DefaultPredictor(cfg)
```


```python
path = "/content/drive/MyDrive/한양도성/detectron2/detectron2_gsm/data/images_test/경복궁_건숙문_건숙문_60.JPG"

im = cv2.imread(path)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(v.get_image()[:, :, ::-1])
```


    
![png](./img/output_16_0.png)
    



```python
path = "/content/drive/MyDrive/한양도성/detectron2/detectron2_gsm/data/images_test/경복궁_건숙문_건숙문_61.JPG"

im = cv2.imread(path)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(v.get_image()[:, :, ::-1])
```


    
![png](./img/output_17_0.png)
    



```python
path = "/content/drive/MyDrive/한양도성/detectron2/detectron2_gsm/data/images_test/경복궁_건숙문_건숙문_62.JPG"

im = cv2.imread(path)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(v.get_image()[:, :, ::-1])
```


    
![png](./img/output_18_0.png)
    



```python

```
