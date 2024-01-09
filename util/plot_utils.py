import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path, PurePath
from util.box_ops import box_cxcywh_to_xyxy
import itertools
import cv2
import random
VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]

VOC_CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorbike","sheep","train",
    "elephant","bear","zebra","giraffe","truck","person"
]

T2_CLASS_NAMES = [
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","diningtable",
    "pottedplant","backpack","umbrella","handbag",
    "tie","suitcase","microwave","oven","toaster","sink",
    "refrigerator","bed","toilet","sofa"
]

T3_CLASS_NAMES = [
    "frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake"
]

T4_CLASS_NAMES = [
    "laptop","mouse","remote","keyboard","cell phone","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush",
    "wine glass","cup","fork","knife","spoon","bowl","tvmonitor","bottle"
]

#old_splits
OLD_VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

OLD_VOC_CLASS_NAMES_COCOFIED = [
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
    "potted plant", "sheep", "couch", "train", "tv"
]

OLD_T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

OLD_T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

OLD_T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]

UNK_CLASS = ["unknown"]

# VOC_COCO_CLASS_NAMES = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))
VOC_COCO_CLASS_NAMES = tuple(itertools.chain(OLD_VOC_CLASS_NAMES, OLD_T2_CLASS_NAMES, OLD_T3_CLASS_NAMES, OLD_T4_CLASS_NAMES, UNK_CLASS))
print(VOC_COCO_CLASS_NAMES)

CLASSES = list(VOC_COCO_CLASS_NAMES)
# colors for visualization
BGR_COLORS = [[250, 51, 153],
                [20, 128, 48],
                [30, 105, 210],
                [255, 0, 0], 
                [192, 192, 192], 
                [15, 94, 56],
                [255,97,0]]
COLORS = [[240, 32, 160],
                [34, 139, 34],
                [0, 215, 255],
                [205, 90, 106], 
                [192, 192, 192], 
                [15, 94, 56],
                [255,97,0]]
TWO_CLORS = [[255, 0, 0],
                [0, 0, 255]]

def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

 

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

 

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

 

    '''
    func_name = "plot_utils.py::plot_logs"

 

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

 

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

 

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if dir.exists():
            continue
        raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")

 

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

 

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

 

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(pd.np.stack(df.test_coco_eval.dropna().values)[:, 1]).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)

 

def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)
    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs

 

def plot_opencv(boxes, output):
    for (x, y, w, h) in boxes:
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    plt.imshow(output)
    plt.show()

 

def plot_image(ax, img, norm):
    if norm:
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
    img = img.astype('uint8')
    ax.imshow(img)

 

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = out_bbox
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(out_bbox)
    return b

def plot_prediction_indices(image, boxes, ax=None, plot_prob=True):    
    if ax is None:
        ax = plt.gca()
    plot_results_indices(image[0].permute(1, 2, 0).detach().cpu().numpy(), boxes, ax, plot_prob=plot_prob)

 

def plot_results_indices(pil_img, boxes, ax, plot_prob=True, norm=True):
    from matplotlib import pyplot as plt
    image = plot_image(ax, pil_img, norm)
    if boxes is not None:
        for (xmin, ymin, xmax, ymax) in boxes.tolist():
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color='r', linewidth=2))
    ax.grid('off')

 

def _plot_prediction_indices(image, boxes, ax=None, plot_prob=True):    
    if ax is None:
        ax = plt.gca()
    _plot_results_indices(image[0].permute(1, 2, 0).detach().cpu().numpy(), boxes, ax, plot_prob=plot_prob)

 

def _plot_results_indices(pil_img, boxes, ax, plot_prob=True, norm=True):
    from matplotlib import pyplot as plt
    image = plot_image(ax, pil_img, norm)
    if boxes is not None:
        for (xmin, ymin, xmax, ymax) in boxes.tolist():
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color='b', linewidth=2))
    ax.grid('off')

 

def plot_prediction(image, scores, boxes, labels, size, ax=None, plot_prob=True):    
    if ax is None:
        ax = plt.gca()
    plot_results(image[0].permute(1, 2, 0).detach().cpu().numpy(), scores, boxes, labels,size, ax, plot_prob=plot_prob)    
    # plot_results(image.permute(1, 2, 0).detach().cpu().numpy(), scores, boxes, labels,size, ax, plot_prob=plot_prob)

def plot_visual(img, scores, boxes, labels, size, norm=True):
    from matplotlib import pyplot as plt 
    if norm:
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
    img = img.copy().astype('uint8')
    w,h = size[0],size[1]
    # plt.imshow(img)
    # cv2.imshow("123",img)
    colors = COLORS * 100
    boxes = rescale_bboxes(boxes, size)
    sc = scores.cpu().tolist()
    # cl = labels.cpu().tolist()
    cl = torch.tensor(labels,dtype=int).cpu().tolist()
    box = boxes.cpu().tolist()
    # if boxes is not None:
    #     # for sc, cl, (xmin, ymin, xmax, ymax), c in zip(scores.squeeze(0).cpu().numpy(), labels.squeeze(0).cpu().numpy(), boxes.squeeze(0).tolist(), colors):
    #     for i in range(len(box)):  
    #         xmin, ymin, xmax, ymax = box[i]
    #         if xmin == 0 and ymin ==0 and xmax==0 and ymax==0:
    #             continue
    #         else:
    #             text = f'{int(100*sc[i])}%'
    #             # text = f'{CLASSES[cl[i]]}:{int(100*sc[i])}%'
    #             if xmin < 0:
    #                 xmin = 1
    #             if ymin < 0:
    #                 ymin = 1
    #             if xmax > w:
    #                 xmax = w-1
    #             if ymax > h:
    #                 ymax = h - 1
    #             if 'unknown' in text:
    #                 id = 0
    #             else:
    #                 id = 1

    #             blk = np.zeros(img.shape, np.uint8)
    #             cv2.rectangle(blk, (int(xmin), int(ymin)), (int(xmax), int(ymax)), BGR_COLORS[id], 5)   # 注意在 blk的基础上进行绘制；#参数分别代表（xmin，ymin）（xmaxm，ymax）（BGR）(粗细)
    #             img = cv2.addWeighted(img, 1, blk, 1, 1)#第二个参数是透明度
    #             # img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 4 )
    #             text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 2)
    #             text_w, text_h = text_size
    #             cv2.rectangle(img, (int(xmin+5), int(ymin+5)), (int(xmin+5) + text_w, int(ymin+5) + text_h+7), (35,35,35), -1)
    #             img = cv2.putText(img, text, (int(xmin+5),int(ymin+5)+ text_h + 1 - 1),cv2.FONT_HERSHEY_TRIPLEX, 1, (201, 230, 252), 1)        
    # return img[...,::-1]  # rgb --> bgr
    if boxes is not None:
        # for sc, cl, (xmin, ymin, xmax, ymax), c in zip(scores.squeeze(0).cpu().numpy(), labels.squeeze(0).cpu().numpy(), boxes.squeeze(0).tolist(), colors):
        for i in range(len(box)):  
            xmin, ymin, xmax, ymax = box[i]
            if xmin == 0 and ymin ==0 and xmax==0 and ymax==0:
                continue
            else:
                text = f'{CLASSES[cl[i]]}:{int(100*sc[i])}%'
                #text = f'{int(100*sc[i])}%'
                if xmin < 0:
                    xmin = 1
                if ymin < 0:
                    ymin = 1
                if xmax > w:
                    xmax = w-1
                if ymax > h:
                    ymax = h - 1
                if i > 6:
                    i = i % 6
                # if 'unknown' in text:
                #     color = (240,32,160)
                # else:
                #     color = colors[i]
                color1 = int(255*random.random())
                color2 = int(255*random.random())
                color3 = int(255*random.random())
                blk = np.zeros(img.shape, np.uint8)
                cv2.rectangle(blk, (int(xmin), int(ymin)), (int(xmax), int(ymax)), BGR_COLORS[i], 5)   # 注意在 blk的基础上进行绘制；#参数分别代表（xmin，ymin）（xmaxm，ymax）（BGR）(粗细)
                img = cv2.addWeighted(img, 1, blk, 1, 1)#第二个参数是透明度
                # img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 4 )
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 1)
                text_w, text_h = text_size
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmin) + text_w, int(ymin) + text_h+7), (0,0,0), -1)
                img = cv2.putText(img, text, (int(xmin),int(ymin)+ text_h + 1 - 1),cv2.FONT_HERSHEY_TRIPLEX, 1, (201, 230, 252), 1)        
    return img[...,::-1]  # rgb --> bgr


def plot_results(pil_img, scores, boxes, labels, size, ax, plot_prob=True, norm=True):
    from matplotlib import pyplot as plt
    image = plot_image(ax, pil_img, norm)
    colors = COLORS * 100
    boxes = rescale_bboxes(boxes, size)
    sc = scores.cpu().tolist()
    cl = labels.cpu().tolist()
    box = boxes.cpu().tolist()
    if boxes is not None:
        # for sc, cl, (xmin, ymin, xmax, ymax), c in zip(scores.squeeze(0).cpu().numpy(), labels.squeeze(0).cpu().numpy(), boxes.squeeze(0).tolist(), colors):
        for i in range(len(box)):  
            xmin, ymin, xmax, ymax = box[i]
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=colors[i], linewidth=2))
            text = f'{CLASSES[cl[i]]}: {sc[i]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=5, bbox=dict(facecolor='yellow', alpha=0.5))
    ax.grid('off')
 

def plot_prediction_GT(image, boxes, labels, ax=None, plot_prob=True):
    bboxes_scaled0 = rescale_bboxes(boxes, list(image.shape[2:])[::-1])    
    if ax is None:
        ax = plt.gca()
    plot_results_GT(image[0].permute(1, 2, 0).detach().cpu().numpy(), bboxes_scaled0, labels, ax, plot_prob=plot_prob)
 

def plot_results_GT(pil_img, boxes, labels, ax, plot_prob=True, norm=True):
    from matplotlib import pyplot as plt
    image = plot_image(ax, pil_img, norm)
    colors = COLORS * 100
    cl = labels.cpu().tolist()
    box = boxes[0].tolist()
    if boxes is not None:
        # for cl, (xmin, ymin, xmax, ymax), c in zip(labels, boxes.squeeze(0).tolist(), colors):
        for i in range(len(box)):
            xmin, ymin, xmax, ymax = box[i]
            c = colors[i]
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=2))
            
            text = f'{CLASSES[cl[i]]}'
            ax.text(xmin, ymin, text, fontsize=5, bbox=dict(facecolor='yellow', alpha=0.5))
    ax.grid('off')



