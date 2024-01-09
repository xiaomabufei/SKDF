from curses.ascii import SP
from dis import dis
import imp
from this import d
from tkinter import S
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from math import *
class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
class Tangent_domain_cluster_loss(_Loss):
    __constants__ = ['margin', 'reduction']
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(Tangent_domain_cluster_loss, self).__init__(size_average, reduce, reduction)
    def cluster(self, distance, Sparse_Radius):
        # loss = distance - Sparse_Radius
        # loss[loss < 0] = 0
        loss = distance.mean()
        return loss
    def tangent(self, distance, Dynamic_Interval):
        # mask = torch.gt(distance, Dynamic_Interval)
        # #dis1 <= Dynamic_Interval, dis2 > Dynamic_Interval
        # dis1, dis2 = torch.masked_select(distance, ~mask), torch.masked_select(distance, mask)
        # pai = torch.tensor(pi, device=Dynamic_Interval.device)
        # dis1_loss = -torch.tan(pai*dis1/(2*Dynamic_Interval)-pai/2)
        # dis2_loss = torch.exp((dis2-Dynamic_Interval)/1000)-1
        # loss = torch.cat([dis1_loss, dis2_loss], dim=0)
        loss = distance.mean()
        return loss   

    def forward(self,batch_size, input, cls_target, fore_target, Dynamic_Interval, Sparse_Radius):
        # jiang lei nei he lei jian te zheng fen kai; lei nei y = 1; lei jian y = -1; 
        #1、 ge lei zhi jian ju lei  2、qian jing yu bei jing ju lei
        cls_mask = cls_target == 1
        fore_mask = fore_target == 1
        mask = cls_target == 0
        cls_distance, fore_distance = torch.masked_select(input, cls_mask), torch.masked_select(input, fore_mask)
        # loss = intraclass_distance  
        cluster_loss = self.cluster(cls_distance, Sparse_Radius)
        # tangent_loss = self.tangent(fore_distance, Dynamic_Interval)
        loss = cluster_loss
        # loss = torch.cat([cluster_loss, tangent_loss], dim=0)
        return loss
