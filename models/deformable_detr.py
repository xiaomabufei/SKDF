# ------------------------------------------------------------------------
# OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
from doctest import FAIL_FAST
import re
import time
from turtle import st
import selectivesearch
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import pickle
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss) #, sigmoid_focal_loss_CA)
from .deformable_transformer import build_deforamble_transformer
from .pseduo_label import find_top_rpn_proposals, add_ground_truth_to_proposals, add_ground_truth_to_proposals_single_image
import copy
import heapq
import operator
import os
from copy import deepcopy
from util.store import Store, Memory_loss
from util.events import get_event_storage
import util.comm as comm
from .IOU_score import bbox_overlaps
from .box_coder import bboxes2tblr
from .L1loss import L1Loss
from util.box_ops import box_cxcywh_to_xyxy


# from .tangent_domain_cluster import Tangent_domain_cluster_loss
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, 
                 unmatched_boxes=False, novelty_cls=False, featdim=1024):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        # self.local_quality_embed = nn.Linear(hidden_dim, 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.featdim = featdim
        self.unmatched_boxes = unmatched_boxes
        self.novelty_cls = novelty_cls
        if self.novelty_cls:
            self.nc_class_embed = nn.Linear(hidden_dim, 1)

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        # self.local_quality_embed.bias.data = torch.ones(1) * bias_value
        if self.novelty_cls:
            self.nc_class_embed.bias.data = torch.ones(1) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.local_decoder.num_layers + 1) if two_stage else transformer.local_decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.local_decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])

            # self.local_quality_embed = nn.ModuleList([self.local_quality_embed for _ in range(num_pred)])
            if self.novelty_cls:
                self.nc_class_embed = nn.ModuleList([self.nc_class_embed for _ in range(num_pred)])

            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.local_decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.class_decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
    def get_feature_map(self, heat_map,h,w):
        heat_map_mean = torch.mean(heat_map, dim=1)
        # feature_map_sum = BI.transform(feature_map_sum)
        map = []
        for i in range(heat_map.shape[0]):
            map.append(F.interpolate(
                        heat_map_mean[i].unsqueeze(0).unsqueeze(0),
                        size = [w, h],
                        mode="nearest",
                    ))
        return torch.cat(map, dim=0)
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        samples_tensor, _ = NestedTensor.decompose(samples)
        _,_,h,w = samples_tensor.size()
        features, pos = self.backbone(samples)
        # print('backbone time', end-start)
        srcs = []
        masks = []
        heat_map = {}
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            ## [Info] extracting the resnet features which are used for selecting unmatched queries
            if self.unmatched_boxes:
                heat_map[f'layer_{l}']= self.get_feature_map(src.clone(), h, w)
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        
        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
        local_hs, class_hs, init_reference, local_inter_references, memory = self.transformer(srcs, masks, pos, query_embeds)
        # print('transformer time', end - start)
        outputs_classes = []
        outputs_coords = []
        outputs_classes_nc = []
        # outputs_coords_scores = []
        for lvl in range(local_hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = local_inter_references[lvl - 1]
            
            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](local_hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            if self.novelty_cls:
                outputs_class_nc = self.nc_class_embed[lvl](local_hs[lvl])
            if self.novelty_cls:
                outputs_classes_nc.append(outputs_class_nc)
            outputs_coord = tmp.sigmoid()
            # box_quality = self.local_quality_embed[lvl](local_hs[lvl])
            outputs_coords.append(outputs_coord)
            # outputs_coords_scores.append(box_quality)
        if self.novelty_cls:
            output_class_nc = torch.stack(outputs_classes_nc)
        outputs_coord = torch.stack(outputs_coords)
        # outputs_coords_score = torch.stack(outputs_coords_scores)
        for lvl in range(class_hs.shape[0]):
            outputs_class = self.class_embed[lvl](class_hs[lvl])
            ## novelty classification
            outputs_classes.append(outputs_class)
            
        outputs_class = torch.stack(outputs_classes)
        
            
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'heat_map': heat_map}

        if self.novelty_cls:
            out = {'pred_logits': outputs_class[-1], 'pred_nc_logits': output_class_nc[-1], 'pred_boxes': outputs_coord[-1],
                     'heat_map': heat_map}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord,  output_class_nc=None)
            if self.novelty_cls:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord,  output_class_nc=output_class_nc)
        # if self.two_stage:
        #     enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        #     out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_class_nc=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # import pdb;pdb.set_trace()
        if output_class_nc is not None:
            xx = [{'pred_logits': a, 'pred_nc_logits': c, 'pred_boxes': b}
                for a, c, b in zip(outputs_class[:-1], output_class_nc[:-1], outputs_coord[:-1])]
        else:
            xx = [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return xx
def weight1(epoch):
    return 1.2-np.exp(-epoch/170)
def weight2(epoch):
    return np.exp(-epoch/170)-0.2
def unnormalize(tensor, mean, std, inplace: bool = False):
    """Unnormalize a tensor image with mean and standard deviation.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]
    # inter[:, :, 0] is the width of intersection and inter[:, :, 1] is height
def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [A,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [B,4]
    Return:
        jaccard overlap: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.nc_epoch = args.nc_epoch
        self.output_dir = args.output_dir
        self.invalid_cls_logits = invalid_cls_logits
        self.unmatched_boxes = args.unmatched_boxes
        self.top_unk = args.top_unk
        self.bbox_thresh = args.bbox_thresh
        self.num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
        self.softmax = nn.Softmax()
        self.pseudo = args.pseudo
        self.objectness = args.objectness
        self.objectness_weights = args.objectness_weights
        self.loss_objectness = L1Loss()
        self.normalizer = 4.0
    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes`` in the (top, left,
        bottom, right) order.

        Args:
            bboxes (torch.Tensor): source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): target of the transformation, e.g.,
                ground truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bboxes2tblr(
            bboxes, gt_bboxes, normalizer=self.normalizer)
        return encoded_bboxes

    def loss_NC_labels(self, outputs, targets, indices, num_boxes, glip_targets, glip_indices, owod_targets, owod_indices, log=True):
        """Novelty classification loss
        target labels will contain class as 1
        owod_indices -> indices combining matched indices + psuedo labeled indices
        owod_targets -> targets combining GT targets + psuedo labeled unknown targets
        target_classes_o -> contains all 1's
        """
        assert 'pred_nc_logits' in outputs
        src_logits = outputs['pred_nc_logits']
        idx = self._get_src_permutation_idx(owod_indices)

        # cimputing objectness loss
        if self.objectness == 'no':
            losses = {'loss_NC': torch.tensor(0,dtype=torch.float32,device=src_logits.device)}
        elif self.objectness == 'object_score':
            target_classes_o = torch.cat([torch.full_like(t["labels"][J], 0) for t, (_, J) in zip(owod_targets, owod_indices)])
            target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target_confidence_o = torch.cat([t["confidence"][J] for t, (_, J) in zip(owod_targets, owod_indices)])
            target_confidence = torch.full(src_logits.shape[:2], 0, dtype=torch.float32, device=src_logits.device)
            target_confidence[idx] = target_confidence_o
            target_confidence = target_confidence.unsqueeze(-1)
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:,:,:-1] * target_confidence
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
            losses = {'loss_NC': loss_ce}
        elif self.objectness == 'IOU':
            #computing IOU score
            prediction_bboxes =  outputs['pred_boxes'][idx]
            gt_bboxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(owod_targets, owod_indices)], dim=0)
            pos_objectness_targets = bbox_overlaps(
                    box_cxcywh_to_xyxy(prediction_bboxes).unsqueeze(0),
                    box_cxcywh_to_xyxy(gt_bboxes).unsqueeze(0),
                    is_aligned=True)
            #computing loss
            objectness = src_logits[idx].squeeze(-1)
            loss_ce = self.loss_objectness(objectness, pos_objectness_targets.squeeze(), self.objectness_weights, avg_factor=num_boxes,reduction_override=None).mean()
            losses = {'loss_NC': loss_ce}
        elif self.objectness == 'centerness':
            pos_bboxes =  outputs['pred_boxes'][idx]
            pos_gt_bboxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(owod_targets, owod_indices)], dim=0)
            pos_objectness_bbox_targets = self.encode(box_cxcywh_to_xyxy(pos_bboxes), box_cxcywh_to_xyxy(pos_gt_bboxes))
            valid_targets = torch.min(pos_objectness_bbox_targets,-1)[0] > 0
            pos_objectness_bbox_targets[valid_targets==False,:] = 0
            top_bottom = pos_objectness_bbox_targets[:,0:2]
            left_right = pos_objectness_bbox_targets[:,2:4]
            pos_objectness_targets = torch.sqrt(
                (torch.min(top_bottom, -1)[0] / 
                    (torch.max(top_bottom, -1)[0] + 1e-12)) *
                (torch.min(left_right, -1)[0] / 
                    (torch.max(left_right, -1)[0] + 1e-12)))
            objectness = src_logits[idx].squeeze(-1)
            loss_ce = self.loss_objectness(
            objectness, 
            pos_objectness_targets, 
            self.objectness_weights, 
            avg_factor=num_boxes).mean()
            losses = {'loss_NC': loss_ce}
        elif self.objectness == 'ours':
            prediction_bboxes =  outputs['pred_boxes'][idx]
            gt_bboxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(owod_targets, owod_indices)], dim=0)
            losses = {'loss_NC': torch.tensor(0,dtype=torch.float32,device=src_logits.device)}
        else:
            raise ValueError(
                    'objectness_type must be in the choices [no, object_score, IOU, centerness, ours].')
        return losses
    def loss_labels(self, outputs, targets, indices, num_boxes, glip_targets, glip_indices, owod_targets, owod_indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        ## comment lines from 317-320 when running for oracle settings
        temp_src_logits = outputs['pred_logits'].clone()
        temp_src_logits[:,:, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits

        if self.unmatched_boxes:
            idx = self._get_src_permutation_idx(owod_indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(owod_targets, owod_indices)])
            target_confidence_o = torch.cat([t["confidence"][J] for t, (_, J) in zip(owod_targets, owod_indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
            target_confidence = torch.full(src_logits.shape[:2], 0, dtype=torch.float32, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target_confidence[idx] = target_confidence_o
            target_confidence = target_confidence.unsqueeze(-1).repeat(1,1,src_logits.shape[-1])
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1] * target_confidence
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
            losses = {'loss_ce': loss_ce}
        else:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
            losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, glip_targets, glip_indices, owod_targets, owod_indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        temp_pred_logits = outputs['pred_logits'].clone()
        temp_pred_logits[:,:, self.invalid_cls_logits] = -10e10
        pred_logits = temp_pred_logits
     
        device = pred_logits.device
        if self.unmatched_boxes:
            tgt_lengths = torch.as_tensor([len(v["labels"]) for v in owod_targets], device=device)
        else:
            tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, glip_targets, glip_indices, owod_targets, owod_indices):
    # def loss_boxes(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices, ca_owod_targets, ca_owod_indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """

        assert 'pred_boxes' in outputs
        
        if self.unmatched_boxes:
            if not self.pseudo:
                idx = self._get_src_permutation_idx(owod_indices)
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(owod_targets, owod_indices)], dim=0)
                target_confidence_o = torch.cat([t["confidence"][J] for t, (_, J) in zip(owod_targets, owod_indices)])
                src_boxes = outputs['pred_boxes'][idx]
                loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
                target_confidence = target_confidence_o.unsqueeze(-1).repeat(1,loss_bbox.shape[-1])
                loss_bbox = loss_bbox * target_confidence
                losses = {}
                losses['loss_bbox'] = loss_bbox.sum() / num_boxes
                loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
                loss_giou = loss_giou * target_confidence_o
                losses['loss_giou'] = loss_giou.sum() / num_boxes
            else:
                idx = self._get_src_permutation_idx(glip_indices)
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(glip_targets, glip_indices)], dim=0)
                target_confidence_o = torch.cat([t["confidence"][J] for t, (_, J) in zip(glip_targets, glip_indices)])
                src_boxes = outputs['pred_boxes'][idx]
                loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
                target_confidence = target_confidence_o.unsqueeze(-1).repeat(1,loss_bbox.shape[-1])
                loss_bbox = loss_bbox * target_confidence
                losses = {}
                losses['loss_bbox'] = loss_bbox.sum() / num_boxes
                loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
                loss_giou = loss_giou * target_confidence_o
                losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            src_boxes = outputs['pred_boxes'][idx]

            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

            losses = {}
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, glip_targets, glip_indices, owod_targets, owod_indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def save_dict(self, di_, filename_):
        with open(filename_, 'wb') as f:
            pickle.dump(di_, f)

    def load_dict(self, filename_):
        with open(filename_, 'rb') as f:
            ret_dict = pickle.load(f)
        return ret_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_single_permutation_idx(self, indices, index):
        ## Only need the src query index selection from this function for attention feature selection
        batch_idx = [torch.full_like(src, i) for i, src in enumerate(indices)][0]
        src_idx = indices[0]
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, glip_targets, glip_indices, owod_targets, owod_indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'NC_labels': self.loss_NC_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, glip_targets, glip_indices, owod_targets, owod_indices, **kwargs)

    def forward(self, samples, outputs, targets, epoch, Iter, last_loss):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        batch = outputs_without_aux['pred_boxes'].shape[0]
        indices = self.matcher(outputs_without_aux, targets, switch=False)
        glip_targets = deepcopy(targets)
        owod_outputs = outputs_without_aux.copy()
        owod_device = owod_outputs["pred_boxes"].device
        #pseudo labels setting
        # 1. merging directly
        for i in range(len(targets)):
            if 'unknown_boxes' in targets[i]:
                unk_label = torch.as_tensor([self.num_classes-1], device=owod_device)
                unk_len = targets[i]['unknown_boxes'].shape[0]
                glip_targets[i]['confidence'] = torch.cat((torch.ones_like(glip_targets[i]['labels'], device=owod_device, dtype=torch.float32), \
                                                                 glip_targets[i]['unknown_score']), dim=0)
                glip_targets[i]['labels'] = torch.cat((glip_targets[i]['labels'], unk_label.repeat_interleave(unk_len)))
                glip_targets[i]['boxes'] = torch.cat((glip_targets[i]['boxes'], glip_targets[i]['unknown_boxes']), dim=0)
                glip_targets[i].pop('unknown_boxes')
                glip_targets[i].pop('unknown_score')
            else:
                glip_targets[i]['confidence'] = torch.ones_like(glip_targets[i]['labels'], device=owod_device, dtype=torch.float32)
        glip_indices = self.matcher(outputs_without_aux, glip_targets,switch=True)

        # pseudo labeling
        owod_targets = deepcopy(glip_targets)
        owod_indices = deepcopy(glip_indices)
        ####wei wan cheng
        if self.unmatched_boxes and epoch >= self.nc_epoch and self.pseudo:
            ## get pseudo unmatched boxes from this section
            queries = torch.arange(outputs['pred_logits'].shape[1])
            object_score = F.softmax(owod_outputs['pred_nc_logits'].view(batch, -1), dim=-1)
            for i in range(len(glip_indices)):
                combined = torch.cat((queries, self._get_src_single_permutation_idx(glip_indices[i], i)[-1])) ## need to fix the indexing
                uniques, counts = combined.unique(return_counts=True)
                unmatched_indices = uniques[counts == 1]
                means_bb = torch.ones(queries.shape[0]).to(owod_device) * -5
                means_bb[unmatched_indices] = object_score[i][unmatched_indices]
                _, topk_inds =  torch.topk(means_bb, self.top_unk)
                topk_inds = torch.as_tensor(topk_inds)
                topk_inds = topk_inds.cpu()
                unk_label = torch.as_tensor([self.num_classes], device=owod_device)
                owod_targets[i]['labels'] = torch.cat((owod_targets[i]['labels'], unk_label.repeat_interleave(self.top_unk)))
                owod_indices[i] = (torch.cat((owod_indices[i][0], topk_inds)), torch.cat((owod_indices[i][1],\
                     (owod_targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu())))
                owod_targets[i]['labels'][owod_targets[i]['labels']==unk_label] = unk_label - 1
                owod_targets[i]['confidence'] = torch.cat((owod_targets[i]['confidence'], object_score[i][topk_inds]))

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in glip_targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, glip_targets, glip_indices, owod_targets, owod_indices, **kwargs))
        # print('match time', end-start)
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for p, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, switch=False)
                glip_targets = deepcopy(targets)
                aux_owod_outputs = aux_outputs.copy()
                owod_device = aux_owod_outputs["pred_boxes"].device
                for i in range(len(targets)):
                    if 'unknown_boxes' in targets[i]:
                        unk_label = torch.as_tensor([self.num_classes-1], device=owod_device)
                        unk_len = targets[i]['unknown_boxes'].shape[0]
                        glip_targets[i]['confidence'] = torch.cat((torch.ones_like(glip_targets[i]['labels'], device=owod_device, dtype=torch.float32), \
                                                                 glip_targets[i]['unknown_score']), dim=0)
                        glip_targets[i]['labels'] = torch.cat((glip_targets[i]['labels'], unk_label.repeat_interleave(unk_len)))
                        glip_targets[i]['boxes'] = torch.cat((glip_targets[i]['boxes'], glip_targets[i]['unknown_boxes']), dim=0)
                        glip_targets[i].pop('unknown_boxes')
                        glip_targets[i].pop('unknown_score')
                    else:
                        glip_targets[i]['confidence'] = torch.ones_like(glip_targets[i]['labels'], device=owod_device, dtype=torch.float32)
                glip_indices = self.matcher(aux_owod_outputs, glip_targets,switch=True)
                # pseudo labeling
                owod_targets = deepcopy(glip_targets)
                owod_indices = deepcopy(glip_indices)
                ####wei wan cheng
                if self.unmatched_boxes and epoch >= self.nc_epoch and self.pseudo:
                    ## get pseudo unmatched boxes from this section
                    queries = torch.arange(outputs['pred_logits'].shape[1])
                    object_score = F.softmax(owod_outputs['pred_nc_logits'].view(batch, -1), dim=-1)
                    for i in range(len(glip_indices)):
                        combined = torch.cat((queries, self._get_src_single_permutation_idx(glip_indices[i], i)[-1])) ## need to fix the indexing
                        uniques, counts = combined.unique(return_counts=True)
                        unmatched_indices = uniques[counts == 1]
                        means_bb = torch.ones(queries.shape[0]).to(owod_device) * -5
                        means_bb[unmatched_indices] = object_score[i][unmatched_indices]
                        _, topk_inds =  torch.topk(means_bb, self.top_unk)
                        topk_inds = torch.as_tensor(topk_inds)
                        topk_inds = topk_inds.cpu()
                        unk_label = torch.as_tensor([self.num_classes], device=owod_device)
                        owod_targets[i]['labels'] = torch.cat((owod_targets[i]['labels'], unk_label.repeat_interleave(self.top_unk)))
                        owod_indices[i] = (torch.cat((owod_indices[i][0], topk_inds)), torch.cat((owod_indices[i][1],\
                            (owod_targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu())))
                        owod_targets[i]['labels'][owod_targets[i]['labels']==unk_label] = unk_label - 1
                        owod_targets[i]['confidence'] = torch.cat((owod_targets[i]['confidence'], object_score[i][topk_inds]))
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, glip_targets, glip_indices, owod_targets, owod_indices, **kwargs)
                    l_dict = {k + f'_{p}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(owod_targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, support, KNOWN_CLS):
        super().__init__()
        self.support = False
        self.supportv2 = support
        self.KNOWN_CLS = KNOWN_CLS
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # nc_out_logits, out_logits, out_bbox = outputs['pred_nc_logits'], outputs['pred_logits'], outputs['pred_boxes']
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = args.num_classes
    print(num_classes)
    if args.dataset == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    prev_intro_cls = args.PREV_INTRODUCED_CLS
    curr_intro_cls = args.CUR_INTRODUCED_CLS
    seen_classes = prev_intro_cls + curr_intro_cls
    invalid_cls_logits = list(range(seen_classes, num_classes-1)) #unknown class indx will not be included in the invalid class range
    print("Invalid class rangw: " + str(invalid_cls_logits))

    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        unmatched_boxes=args.unmatched_boxes,
        novelty_cls=args.NC_branch,
        featdim=args.featdim,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    if args.NC_branch:
        weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_NC': args.nc_loss_coef, 'loss_bbox': args.bbox_loss_coef}

    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.local_dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.NC_branch:
        if args.first_stage:
            losses = ['NC_labels', 'boxes', 'cardinality']
        elif args.second_stage:
            losses = ['labels', 'cardinality']
        else:
            losses = ['labels', 'NC_labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, focal_alpha=args.focal_alpha)
    criterion.to(device)
    KNOWN_CLS = args.CUR_INTRODUCED_CLS + args.PREV_INTRODUCED_CLS
    postprocessors = {'bbox': PostProcess(args.support, KNOWN_CLS)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors