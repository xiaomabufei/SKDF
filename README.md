# SKDFSKDFSKDF


# Abstract
Open World Object Detection (OWOD) is a novel computer vision task with a considerable challenge, bridging the gap between classic object detection (OD) benchmarks and real-world object detection. In addition to detecting and classifying seen/known objects, OWOD algorithms are expected to localize all potential unseen/unknown objects and incrementally learn them. The large pre-trained vision-language grounding models (VLM, \eg, GLIP) have rich knowledge about the open world, but are limited by text prompts and cannot localize indescribable objects. However, there are many detection scenarios in which pre-defined language descriptions are unavailable during inference. In this paper, we attempt to specialize the VLM model for OWOD tasks by distilling its open-world knowledge into a language-agnostic detector. Surprisingly, we observe that the combination of a simple knowledge distillation approach and the automatic pseudo-labeling mechanism in OWOD can achieve better performance for unknown object detection, even with a small amount of data. Unfortunately, knowledge distillation for unknown objects severely affects the learning of detectors with conventional structures for known objects, leading to catastrophic forgetting. To alleviate these problems, we propose the down-weight loss function for knowledge distillation from vision-language to single vision modality. Meanwhile, we propose the cascade decouple decoding structure that decouples the learning of localization and recognition to reduce the impact of category interactions of known and unknown objects on the localization learning process. Ablation experiments demonstrate that both of them are effective in mitigating the impact of open-world knowledge distillation on the learning of known objects. Additionally, to alleviate the current lack of comprehensive benchmarks for evaluating the ability of the open-world detector to detect unknown objects in the open world, we propose two benchmarks, which we name StandardSet and IntensiveSet respectively, based on the complexity of their testing scenarios. Comprehensive experiments performed on OWOD, MS-COCO, and our proposed benchmarks demonstrate the effectiveness of our methods.


![image](https://github.com/xiaomabufei/SKDF/assets/104605826/f86ee584-837c-4354-a5d7-78790145336b)



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

[T1-T4 weight](https://pan.baidu.com/s/1o391DDITLPrpZ2dGNwG3lw?pwd=6666)    

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

[T1-T4 weight](https://pan.baidu.com/s/19xsx8pKO2IO-WdrWb6VypA?pwd=8888)

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

**Acknowledgments:**

SKDF builds on previous works code base such as [OWDETR](https://github.com/akshitac8/ow-detr),[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [OWOD](https://github.com/JosephKJ/OWOD). If you found SKDF useful please consider citing these works as well.

