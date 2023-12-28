# DOWB


# Abstract
Open World Object Detection (OWOD) is a novel computer vision task with a considerable challenge, bridging the gap between classic object detection (OD) benchmarks and real-world object detection. In addition to detecting and classifying seen/known objects, OWOD algorithms are expected to localize all potential unseen/unknown objects and incrementally learn them. The large pre-trained vision-language grounding models (VLM, \eg, GLIP) have rich knowledge about the open world, but are limited by text prompts and cannot localize indescribable objects. However, there are many detection scenarios which pre-defined language descriptions are unavailable during inference. In this paper, we attempt to specialize the VLM model for OWOD task by distilling its open-world knowledge into a language-agnostic detector. Surprisingly, we observe that the combination of a simple knowledge distillation approach and the automatic pseudo-labeling mechanism in OWOD can achieve better performance for unknown object detection, even with a small amount of data. Unfortunately, knowledge distillation for unknown objects severely affects the learning of detectors with conventional structures for known objects, leading to catastrophic forgetting. To alleviate these problems, we propose the down-weight loss function for knowledge distillation from vision-language to single vision modality. Meanwhile, we decouple the learning of localization and recognition to reduce the impact of Category interactions of known and unknown objects on the localization learning process. Comprehensive experiments performed on MS-COCO and PASCAL VOC demonstrate the effectiveness of our methods.


![image](https://github.com/xiaomabufei/SKDF/assets/104605826/1b0cf05d-f2fa-444e-96df-61e83fa853a8)


# Installation

### Requirements

We have trained and tested our models on `Ubuntu 16.0`, `CUDA 10.2`, `GCC 5.4`, `Python 3.7`

```bash
conda create -n SKDF python=3.7 pip
conda activate SKDF
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Backbone features

Download the self-supervised backbone from [here](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth) and add in `models` folder.

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```


# Dataset & Results

### OWOD proposed splits
<br>
<p align="center" ><img width='500' src = "https://imgur.com/9bzf3DV.png"></p> 
<br>

### Results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=2>Task1</th>
        <th align="center" colspan=2>Task2</th>
        <th align="center" colspan=2>Task3</th>
        <th align="center" colspan=1>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">mAP</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center">4.9</td>
        <td align="center">56.0</td>
        <td align="center">2.9</td>
        <td align="center">39.4</td>
        <td align="center">3.9</td>
        <td align="center">29.7</td>
        <td align="center">25.3</td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center">7.5</td>
        <td align="center">59.2</td>
        <td align="center">6.2</td>
        <td align="center">42.9</td>
        <td align="center">5.7</td>
        <td align="center">30.8</td>
        <td align="center">27.8</td>
    </tr>
    <tr>
        <td align="left">SKDF</td>
        <td align="center">39.0</td>
        <td align="center">56.8</td>
        <td align="center">36.7</td>
        <td align="center">40.3</td>
        <td align="center">36.1</td>
        <td align="center">30.1</td>
        <td align="center">26.9</td>
    </tr>
</table>

### Weights

[T1 weight](https://drive.google.com/file/d/1oBD-y5M2ptk-cLrLaUpWxdP03Jig4yGk/view?usp=sharing)      |      [T2_ft weight](https://drive.google.com/file/d/1bHYjcJdp67ndcVw6VP_CZ5IY90eaf2HN/view?usp=sharing)      |      [T3_ft weight](https://drive.google.com/file/d/1vhJC9MZU7K9HssihhMQ_BcqIlDBECbn2/view?usp=sharing)      |      [T4_ft weight](https://drive.google.com/file/d/1d766FdmIc07zaWlbW1slCQGZuQmggEM3/view?usp=sharing)

### OWDETR proposed splits

<br>
<p align="center" ><img width='500' src = "https://imgur.com/RlqbheH.png"></p> 
<br>

### Results

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=2>Task1</th>
        <th align="center" colspan=2>Task2</th>
        <th align="center" colspan=2>Task3</th>
        <th align="center" colspan=1>Task4</th>
    </tr>
    <tr>
        <td align="left">Method</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">U-Recall</td>
        <td align="center">mAP</td>
        <td align="center">mAP</td>
    </tr>
    <tr>
        <td align="left">ORE-EBUI</td>
        <td align="center">1.5</td>
        <td align="center">61.4</td>
        <td align="center">3.9</td>
        <td align="center">40.6</td>
        <td align="center">3.6</td>
        <td align="center">33.7</td>
        <td align="center">31.8</td>
    </tr>
    <tr>
        <td align="left">OW-DETR</td>
        <td align="center">5.7</td>
        <td align="center">71.5</td>
        <td align="center">6.2</td>
        <td align="center">43.8</td>
        <td align="center">6.9</td>
        <td align="center">38.5</td>
        <td align="center">33.1</td>
    </tr>
    <tr>
        <td align="left">SKDF</td>
        <td align="center">60.9</td>
        <td align="center">69.4</td>
        <td align="center">60.0</td>
        <td align="center">44.4</td>
        <td align="center">58.6</td>
        <td align="center">40.1</td>
        <td align="center">39.7</td>
    </tr>
</table>

### Weights

[T1 weight](https://drive.google.com/file/d/1Q9e2bhZ3-VvuOrEGN71SHUBwwhc6qS8q/view?usp=sharing)      |      [T2_ft weight](https://drive.google.com/file/d/1XLOuVnAW5z7Eo_13iy-Ri_SzkcO_2gok/view?usp=sharing)      |      [T3_ft weight](https://drive.google.com/file/d/1XDMU2uulDUFGFyV0UIrU8D9HVBGeDTbh/view?usp=sharing)      |      [T4_ft weight](https://drive.google.com/file/d/1bq5yqwNKKgXZiRrfonda7mNIWislmmW7/view?usp=sharing)

### Dataset Preparation

The splits are present inside `data/VOC2007/SKDF/ImageSets/` folder.
1. Make empty `JPEGImages` and `Annotations` directory.
```
mkdir data/VOC2007/SKDF/JPEGImages/
mkdir data/VOC2007/SKDF/Annotations_selective/
```
2. Download the COCO Images and Annotations from [coco dataset](https://cocodataset.org/#download).
3. Unzip train2017 and val2017 folder. The current directory structure should look like:
```
SKDF/
└── data/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
```
4. Move all images from `train2017/` and `val2017/` to `JPEGImages` folder.
```
cd SKDF/data
mv data/coco/train2017/*.jpg data/VOC2007/SKDF/JPEGImages/.
mv data/coco/val2017/*.jpg data/VOC2007/SKDF/JPEGImages/.
```
5. **Annotations_selective** :The Annotations can be made by the file **"make_pseudo_labels.py"**

The files should be organized in the following structure:
```
SKDF/
└── data/
    └── VOC2007/
        └── OWOD/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations_selective
```


Currently, Dataloader and Evaluator followed for SKDF is in VOC format.



    
# Training

#### Training on single node

To train SKDF on a single node with 8 GPUS, run
```bash
./run.sh
```

#### Training on slurm cluster

To train SKDF on a slurm cluster having 2 nodes with 8 GPUS each, run
```bash
sbatch run_slurm.sh
```

# Evaluation

For reproducing any of the above mentioned results please run the `run_eval.sh` file and add pretrained weights accordingly.


**Note:**
For more training and evaluation details please check the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) reposistory.

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.




# Contact

Should you have any question, please contact :e-mail: xiaomabufei@gmail.com

**Acknowledgments:**

SKDF builds on previous works code base such as [OWDETR](https://github.com/akshitac8/ow-detr),[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [OWOD](https://github.com/JosephKJ/OWOD). If you found SKDF useful please consider citing these works as well.

