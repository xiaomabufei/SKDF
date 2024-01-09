# partly taken from https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
import functools
from tkinter.messagebox import NO
import torch

import os
import tarfile
import collections
import logging
import copy
from torchvision.datasets import VisionDataset
import itertools

import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
from torch import nn
#new_splits
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

coco_lvis_categories = {
        "person": ["person"], 
        "bicycle": ["bicycle"], 
        "car": ["car_(automobile)", "jeep", "limousine", "minivan", "race_car", "cab_(taxi)", "camper_(vehicle)", "minivan", "fire_engine", "police_cruiser", "ambulance"], 
        "motorbike": ["motorcycle", "motor_scooter", "dirt_bike", "motor_vehicle"], 
        "aeroplane": ["airplane", "jet_plane", "seaplane", "space_shuttle"], 
        "bus": ["bus_(vehicle)", "school_bus"], 
        "train": ["train_(railroad_vehicle)", "bullet_train", "railcar_(part_of_a_train)", "passenger_car_(part_of_a_train)", "freight_car"], 
        "truck": ["truck", "trailer_truck", "tow_truck", "pickup_truck", "garbage_truck"], 
        "boat": ["boat", "gondola_(boat)", "houseboat", "river_boat", "cruise_ship", "cargo_ship", "passenger_ship", "dinghy", "barge", "ferry"], 
        "traffic light": ["traffic_light"], 
        "fire hydrant": ["fireplug"], 
        "stop sign": ["stop_sign", "street_sign"], 
        "parking meter": ["parking_meter"], 
        "bench": ["bench", "pew_(church_bench)"], 
        "bird": ["bird", "hummingbird", "seabird", "flamingo", "owl", "eagle"], 
        "cat": ["cat", "kitten"], 
        "dog": ["dog", "bulldog", "pug-dog", "shepherd_dog", "dalmatian", "puppy"], 
        "horse": ["horse", "pony"], 
        "sheep": ["sheep", "black_sheep", "lamb_(animal)", "goat", "ram_(animal)"], 
        "cow": ["cow", "calf", "bull"], 
        "elephant": ["elephant"], 
        "bear": ["bear", "polar_bear", "grizzly"], 
        "zebra": ["zebra"], 
        "giraffe": ["giraffe"], 
        "backpack": ["backpack", "satchel"], 
        "umbrella": ["umbrella", "parasol"], 
        "handbag": ["handbag", "plastic_bag", "shopping_bag", "shoulder_bag", "tote_bag", "saddlebag", "grocery_bag", "clutch_bag", "sleeping_bag"], 
        "tie": ["necktie", "bolo_tie", "bow-tie"], 
        "suitcase": ["suitcase", "briefcase", "trunk"], 
        "frisbee": ["frisbee"], 
        "skis": ["ski"], 
        "snowboard": ["snowboard"], 
        "sports ball": ["baseball", "basketball", "benchball", "bowling_ball", "football_(American)", "ping-pong_ball", "soccer_ball", "softball", "tennis_ball", "volleyball"], 
        "kite": ["kite"], 
        "baseball bat": ["baseball_bat"], 
        "baseball glove": ["baseball_glove"], 
        "skateboard": ["skateboard"], 
        "surfboard": ["surfboard", "water_ski"], 
        "tennis racket": ["tennis_racket", "racket"], 
        "bottle": ["bottle", "beer_bottle", "wine_bottle", "water_bottle", "thermos_bottle", "jar", "saltshaker"], 
        "wine glass": ["wineglass", "shot_glass", "glass_(drink_container)", "flute_glass", "chalice"], 
        "cup": ["cup", "Dixie_cup", "measuring_cup", "teacup", "mug"], 
        "fork": ["fork", "pitchfork"], 
        "knife": ["knife", "pocketknife", "steak_knife"],
        "spoon": ["spoon", "soupspoon", "wooden_spoon"], 
        "bowl": ["bowl", "soup_bowl", "sugar_bowl", "fishbowl"], 
        "banana": ["banana"], 
        "apple": ["apple"], 
        "sandwich": ["sandwich"], 
        "orange": ["orange_(fruit)", "mandarin_orange", "lime"], 
        "broccoli": ["broccoli", "cauliflower"], 
        "carrot": ["carrot", "turnip", "radish"], 
        "hot dog": ["sausage", "bread"], 
        "pizza": ["pizza", "pie"], 
        "donut": ["doughnut", "bagel"], 
        "cake": ["cake", "birthday_cake", "chocolate_cake", "cupcake", "pancake", "wedding_cake"],
        "chair": ["chair", "armchair", "deck_chair", "folding_chair", "highchair", "rocking_chair", "electric_chair"], 
        "sofa": ["sofa", "sofa_bed", "chaise_longue", "recliner"], 
        "pottedplant": ["flowerpot", "bouquet", "flower_arrangement", "sunflower"], 
        "bed": ["bed", "bunk_bed", "crib"], 
        "diningtable": ["dining_table", "coffee_table", "kitchen_table", "table"], 
        "toilet": ["toilet"], 
        "tvmonitor": ["television_set"], 
        "laptop": ["laptop_computer"], 
        "mouse": ["mouse_(computer_equipment)"], 
        "remote": ["remote_control"], 
        "keyboard": ["computer_keyboard"], 
        "cell phone": ["cellular_telephone"], 
        "microwave": ["microwave_oven"], 
        "oven": ["oven", "toaster_oven"], 
        "toaster": ["toaster"], 
        "sink": ["sink", "kitchen_sink"], 
        "refrigerator": ["refrigerator"], 
        "book": ["book", "booklet", "comic_book", "phonebook", "checkbook", "paperback_book", "hardback_book", "notebook"], 
        "clock": ["clock", "alarm_clock", "wall_clock"], 
        "vase": ["vase"], 
        "scissors": ["scissors", "shears", "clippers_(for_plants)"], 
        "teddy bear": ["teddy bear"], 
        "hair drier": ["hair_dryer"], 
        "toothbrush": ["toothbrush"]}
# VOC_COCO_CLASS_NAMES = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))
VOC_COCO_CLASS_NAMES = tuple(itertools.chain(OLD_VOC_CLASS_NAMES, OLD_T2_CLASS_NAMES, OLD_T3_CLASS_NAMES, OLD_T4_CLASS_NAMES, UNK_CLASS))
print(VOC_COCO_CLASS_NAMES)

class OWDetection(VisionDataset):
    """`OWOD in Pascal VOC format <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 args,
                 root,
                 years='2012',
                 image_sets='train',
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 no_cats=False,
                 filter_pct=-1):
        super(OWDetection, self).__init__(root, transforms, transform, target_transform)
        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set = []

        self.CLASS_NAMES = VOC_COCO_CLASS_NAMES
        self.MAX_NUM_OBJECTS = 64
        self.no_cats = no_cats
        self.args = args
        # import pdb;pdb.set_trace()

        for year, image_set in zip(years, image_sets):

            if year == "2007" and image_set == "test":
                year = "2007-test"
            valid_sets = ["t1_train", "t2_train", "t3_train", "t4_train", "t2_ft", "t3_ft", "t4_ft", "test", "all_task_test"]
            if year == "2007-test":
                valid_sets.append("test")

            # base_dir = DATASET_YEAR_DICT[year]['base_dir']
            voc_root = self.root
            if self.args.use_sam:
                annotation_dir = os.path.join(voc_root, 'sam_annotation_fixed')  
            else:
                annotation_dir = os.path.join(voc_root, 'Annotations')   

            print(annotation_dir)
            image_dir = os.path.join(voc_root, 'JPEGImages')

            if not os.path.isdir(voc_root):
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')
            file_names = self.extract_fns(image_set, voc_root)
            self.image_set.extend(file_names)

            self.images.extend([os.path.join(image_dir, x + ".jpg") for x in file_names])
            self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])

            self.imgids.extend(self.convert_image_id(x, to_integer=True) for x in file_names)
            self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        if filter_pct > 0:
            num_keep = float(len(self.imgids)) * filter_pct
            keep = np.random.choice(np.arange(len(self.imgids)), size=round(num_keep), replace=False).tolist()
            flt = lambda l: [l[i] for i in keep]
            self.image_set, self.images, self.annotations, self.imgids = map(flt, [self.image_set, self.images,
                                                                                   self.annotations, self.imgids])
        assert (len(self.images) == len(self.annotations) == len(self.imgids))

    @staticmethod
    def convert_image_id(img_id, to_integer=False, to_string=False, prefix='2021'):
        if to_integer:
            return int(prefix + img_id.replace('_', ''))
        if to_string:
            x = str(img_id)
            assert x.startswith(prefix)
            x = x[len(prefix):]
            if len(x) == 12 or len(x) == 6:
                return x
            return x[:4] + '_' + x[4:]

    @functools.lru_cache(maxsize=None)
    def load_instances(self, img_id):
        tree = ET.parse(self.imgid2annotations[img_id])
        target = self.parse_voc_xml(tree.getroot())
        image_id = target['annotation']['filename']
        h, w = float(target['annotation']['size']['height']), float(target['annotation']['size']['width'])
        instances = []
        pseudo_instances = []
        unknown_instances = [] 

        for obj in target['annotation']['object']:
            cls = obj["name"]
            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            bbox = obj["bndbox"]
            bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            if cls == 'sam':
                cls = 'unknown'
                unknown_instance = dict(
                        category_id=self.args.num_classes - 1,
                        score=1.0,
                        bbox=bbox,
                        area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        image_id=img_id
                    )
                unknown_instances.append(unknown_instance)
            else:
                # #########
                if cls not in VOC_COCO_CLASS_NAMES:
                    cls = 'unknown'
                instance = dict(
                    category_id=VOC_COCO_CLASS_NAMES.index(cls),
                    score=1.0,
                    bbox=bbox,
                    area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    image_id=img_id
                )
                instances.append(instance)

        if 'pseudo_object' in target['annotation']:
            if isinstance(target['annotation']['pseudo_object'], dict):
                pseudo_object_list = [target['annotation']['pseudo_object']]
            else: 
                pseudo_object_list = target['annotation']['pseudo_object']
            for pseudo_obj in pseudo_object_list:
                try:
                    cls = pseudo_obj['name'].split('_')[1]
                except:
                    print(img_id)

                if cls in VOC_CLASS_NAMES_COCOFIED:
                    cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]


                ##############################
                clsincoco = False
                if cls not in VOC_COCO_CLASS_NAMES :
                    for k, v in coco_lvis_categories.items():
                        if cls in v:
                            cls = k
                            clsincoco = True
                            break
                    if clsincoco != True:
                        cls = 'unknown'        
                ##############################
                bbox = pseudo_obj["bndbox"]
                bbox = [float(bbox['xmin'])*w, float(bbox['ymin'])*h, float(bbox['xmax'])*w, float(bbox['ymax'])*h]
                # bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
                # re_bbox = [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                if cls == 'unknown':
                    unknown_instance = dict(
                        category_id=1204,
                        score=float(pseudo_obj['score']),
                        bbox=bbox,
                        area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        image_id=img_id
                    )
                    unknown_instances.append(unknown_instance)
                else:
                    pseudo_instance = dict(
                        category_id=VOC_COCO_CLASS_NAMES.index(cls),
                        score = float(pseudo_obj['score']),
                        bbox=bbox,
                        area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        image_id=img_id
                    )
                    pseudo_instances.append(pseudo_instance)

        return target, instances, pseudo_instances, unknown_instances

    def extract_fns(self, image_set, voc_root):
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        return file_names

    ### OWOD
    def split_kn_unk_instances(self, target):
        # For training data. Removing earlier seen class objects and the unknown objects..
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        unknown_classes = range(prev_intro_cls + curr_intro_cls, self.args.num_classes -1)
        entry = copy.copy(target)
        entry_pseudo = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
            if annotation["category_id"] not in unknown_classes:
                entry_pseudo.remove(annotation)
        return entry, entry_pseudo


    def remove_prev_class_and_unk_instances(self, target):
        # For training data. Removing earlier seen class objects and the unknown objects..
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def remove_unknown_instances(self, target):
        # For finetune data. Removing the unknown objects...
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def label_known_class_and_unknown(self, target):
        # For test and validation data.
        # Label known instances the corresponding label and unknown instances as unknown.
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        total_num_class = self.args.num_classes #81
        known_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in  copy.copy(entry):
        # for annotation in entry:
            if annotation["category_id"] not in known_classes:
                annotation["category_id"] = total_num_class - 1
        return entry

    def remove_known_instances(self, target):
        # For pseudo_instances. Removing the known objects..
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] in valid_classes:
                entry.remove(annotation)
        return entry
    def pseudo_to_target(self, target, pseudo_target):
        # jiang pseudo zhong de zhu shi fang fao target li mian 
        # For pseudo_instances. Removing the known objects..
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] in valid_classes:
                entry.remove(annotation)
        return entry


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        image_set = self.transforms[0]
        img = Image.open(self.images[index]).convert('RGB')
        target, instances, pseudo_instances,unknown_instances = self.load_instances(self.imgids[index])
        if 'train' in image_set:
            if self.args.use_sam:
                instances, pseudo_instances = self.split_kn_unk_instances(instances)
            else:
                instances = self.remove_prev_class_and_unk_instances(instances)
                pseudo_instances = self.remove_known_instances(pseudo_instances)
        elif 'test' in image_set:
            instances = self.label_known_class_and_unknown(instances)
        elif 'ft' in image_set:
            instances = self.remove_unknown_instances(instances)
            pseudo_instances = self.remove_known_instances(pseudo_instances)
        if self.args.lvisnococo == False:
            unknown_instances.extend(pseudo_instances)

        w, h = map(target['annotation']['size'].get, ['width', 'height'])

        if len(unknown_instances) == 0:
            target = dict(
                image_id=torch.tensor([self.imgids[index]], dtype=torch.int64),
                labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
                area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
                boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
                orig_size=torch.as_tensor([int(h), int(w)]),
                size=torch.as_tensor([int(h), int(w)]),
                iscrowd=torch.zeros(len(instances), dtype=torch.uint8)
            )
        elif len(unknown_instances) != 0:
            unknown_boxes = torch.as_tensor([i['bbox'] for i in unknown_instances], dtype=torch.float32)
            keep = (unknown_boxes[:, 3] > unknown_boxes[:, 1]) & (unknown_boxes[:, 2] > unknown_boxes[:, 0])
            unknown_boxes = unknown_boxes[keep]
            target = dict(
                image_id=torch.tensor([self.imgids[index]], dtype=torch.int64),
                labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
                area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
                boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
                unknown_score = torch.tensor([i['score'] for i in unknown_instances], dtype=torch.float32),
                unknown_boxes = unknown_boxes,
                orig_size=torch.as_tensor([int(h), int(w)]),
                size=torch.as_tensor([int(h), int(w)]),
                iscrowd=torch.zeros(len(instances), dtype=torch.uint8)
            )
        if self.transforms[-1] is not None:
            img, target = self.transforms[-1](img, target)
        
        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)











