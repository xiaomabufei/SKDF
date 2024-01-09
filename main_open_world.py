# ------------------------------------------------------------------------
# OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
from re import T
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import OWDetection
from engine import evaluate, train_one_epoch, viz
from models import build_model


from torch.utils.tensorboard import SummaryWriter



def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--local_dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--class_dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # objectness manner
    parser.add_argument('--objectness', default='object_score', type=str, choices=['no','object_score','IOU','centerness','ours'],     
                        help="proposed support localization")     
    parser.add_argument('--objectness_weights', default=1.0, type=float,     
                        help="weight of objectness branch")    
    # train manner             
    parser.add_argument('--first_stage', action='store_true',     
                        help="first stage of the stage-wise training for open world")
    parser.add_argument('--second_stage', action='store_true',     
                        help="first stage of the stage-wise training for open world")
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    # dataset parameters
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='exps/OWOD_t1_new_split',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    ## OWOD
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int)
    parser.add_argument('--CUR_INTRODUCED_CLS', default=20, type=int)
    parser.add_argument('--top_unk', default=5, type=int)
    parser.add_argument('--unmatched_boxes',  default=True, action='store_true')
    parser.add_argument('--featdim', default=1024, type=int)
    parser.add_argument('--pretrained', default='', help='initialized from the pre-training model')
    parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')
    parser.add_argument('--train_set', default='t1_train', help='training txt files')
    parser.add_argument('--test_set', default='all_task_test', help='testing txt files')
    parser.add_argument('--NC_branch', action='store_true')
    parser.add_argument('--nc_loss_coef', default=1, type=float)
    parser.add_argument('--invalid_cls_logits', default=False, action='store_true', help='owod setting')
    parser.add_argument('--nc_epoch', default=5, type=int)
    parser.add_argument('--num_classes', default=81, type=int)
    parser.add_argument('--backbone', default='dino_resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dataset', default='owod')
    parser.add_argument('--data_root', default='/mnt/gluster/ssd/datasets/OWOD/', type=str)
    parser.add_argument('--bbox_thresh', default=0.3, type=float)
    parser.add_argument('--pseudo', default=True, action='store_true', help='whether open pseudo')
    parser.add_argument('--use_tensorboard', action='store_true')
    parser.add_argument('--lvisnococo',default=False, action='store_true')  
    parser.add_argument('--use_sam', default=False, action='store_true')
    return parser

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)
    # print(args.enable_adaptive_pseudo)
    # print(args.enable_clustering)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.use_tensorboard:
        writer = SummaryWriter(comment='exps_lvisnococo_new')

    dataset_train, dataset_val = get_datasets(args)
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    # args.clustering_start_epoch = args.epochs - 20
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    # print(model_without_ddp)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    if args.dataset == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.dataset == "coco":
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds = dataset_val

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    # if args.pretrain:
    #     print('Initialized from the pre-training model')
    #     checkpoint = torch.load(args.pretrain, map_location='cpu')
    #     state_dict = checkpoint['model']
    #     for key in list(state_dict.keys()):
    #         if 'class_embed' in key or 'nc_class_embed' in key or 'bbox_embed' in key:
    #             state_dict.pop(key)
    #     # state_dict = checkpoint
    #     msg = model_without_ddp.load_state_dict(state_dict, strict=False)
    #     print(msg)
    #     # args.start_epoch = checkpoint['epoch'] + 1

    if args.pretrain:
        print('Initialized from the pre-training model')
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(msg)
        args.start_epoch = checkpoint['epoch'] + 1

    if args.second_stage:
        for name, par in model_without_ddp.named_parameters():
            if 'class_decoder' in name or 'class_embed' in name:
                par.requires_grad = True
            else:
                par.requires_grad = False

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if (not args.eval and not args.viz and args.dataset in ['coco', 'voc']):
            test_stats, coco_evaluator, metrics = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args
            )
        if args.eval:
            test_stats, coco_evaluator, metrics = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args)
            # if utils.is_main_process() and args.use_tensorboard:
            #     str1 = '0.8: {50: '
            #     str2 = '}, 0.9: '
            #     writer.add_scalar('unknown/WI', np.float32(metrics['WI'][metrics['WI'].index(str1)+10:metrics['WI'].index(str2)]), 49)
            #     writer.add_scalar('unknown/OSE', np.float32(metrics['OSE'].strip('{50:').rstrip('}')), 49)
            #     if 'Prev_AP' in metrics:
            #         writer.add_scalar('mAP/Prev_AP', np.float32(metrics['Prev_AP'].strip('tensor(').rstrip(')')), 49)
            #     writer.add_scalar('mAP/Current_AP', np.float32(metrics['Current_AP'].strip('tensor(').rstrip(')')), 49)
            #     writer.add_scalar('mAP/Both', np.float32(metrics['Both'].strip('tensor(').rstrip(')')), 49)
            #     writer.add_scalar('unknown/U_Recall', np.float32(metrics['U_Recall']), 49)
            #     writer.add_scalar('unknown/Unknown_Precisions50', np.float32(metrics['Unknown_Precisions50']), 49)     
            #     writer.add_scalar('unknown/num_below03', np.float32(metrics['num_below03']), 49)       
            #     writer.add_scalar('new_metric/error_detection_rate', np.float32(metrics['error_detection_rate']), 49)      
            #     writer.add_scalar('new_metric/num_unknown_rest', np.float32(metrics['num_unknown_rest']), 49)
            #     writer.add_scalar('new_metric/unknown_ap', np.float32(metrics['unknown_ap']), 49)       
            #     writer.add_scalar('new_metric/CSE', np.float32(metrics['CSE']), 49)      
            #     writer.add_scalar('new_metric/TP', np.float32(metrics['TP']), 49)       
            #     writer.add_scalar('new_metric/FP', np.float32(metrics['FP']), 49)               

            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
                if utils.is_main_process():
                    with (output_dir / "log_eval.txt").open("a") as f:
                        f.write(json.dumps(metrics) + "\n")
                        return

    if args.viz:
        viz(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.nc_epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            # if args.enable_clustering and epoch >= args.clustering_start_epoch and utils.is_main_process():
            #     feat_store_save_loc = os.path.join(criterion.feature_store_save_loc, f'feat{epoch}.pt')
            #     print('Saving image store at epoch ' + str(epoch) + ' to ' + feat_store_save_loc)
            #     torch.save(criterion.feature_store, feat_store_save_loc)
            # if args.enable_adaptive_pseudo and utils.is_main_process():
            #     loss_store_save_loc = os.path.join(criterion.loss_store_save_loc, f'loss{epoch}.pt')
            #     print('Saving adaptive weight at epoch ' + str(epoch) + ' to ' + loss_store_save_loc)
            #     torch.save(criterion.loss_memory, loss_store_save_loc)
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if args.dataset in ['owod'] and epoch % args.eval_every == 0 and epoch > 0:
            test_stats, coco_evaluator , metrics = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args
            )
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log_eval.txt").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
            if utils.is_main_process() and args.use_tensorboard:
                str1 = '0.8: {50: '
                str2 = '}, 0.9: '
                writer.add_scalar('unknown/WI', np.float32(metrics['WI'][metrics['WI'].index(str1)+10:metrics['WI'].index(str2)]), epoch)
                writer.add_scalar('unknown/OSE', np.float32(metrics['OSE'].strip('{50:').rstrip('}')), epoch)
                if 'Prev_AP' in metrics:
                    writer.add_scalar('mAP/Prev_AP', np.float32(metrics['Prev_AP'].strip('tensor(').rstrip(')')), epoch)
                writer.add_scalar('mAP/Current_AP', np.float32(metrics['Current_AP'].strip('tensor(').rstrip(')')), epoch)
                writer.add_scalar('mAP/Both', np.float32(metrics['Both'].strip('tensor(').rstrip(')')), epoch)
                writer.add_scalar('unknown/U_Recall', np.float32(metrics['U_Recall']), epoch)
                writer.add_scalar('unknown/Unknown_Precisions50', np.float32(metrics['Unknown_Precisions50']), epoch)     
                writer.add_scalar('unknown/num_below03', np.float32(metrics['num_below03']), epoch)       
                writer.add_scalar('new_metric/error_detection_rate', np.float32(metrics['error_detection_rate']), epoch)      
                writer.add_scalar('new_metric/num_unknown_rest', np.float32(metrics['num_unknown_rest']), epoch)
                writer.add_scalar('new_metric/unknown_ap', np.float32(metrics['unknown_ap']), epoch)       
                writer.add_scalar('new_metric/CSE', np.float32(metrics['CSE']), epoch)      
                writer.add_scalar('new_metric/TP', np.float32(metrics['TP']), epoch)       
                writer.add_scalar('new_metric/FP', np.float32(metrics['FP']), epoch)               

                # writer.add_scalar('WsWf/Ws', np.float32(criterion.loss_memory.Ws), epoch)
                # writer.add_scalar('WsWf/Wf', np.float32(criterion.loss_memory.Wf), epoch)

        else:
            test_stats = {}


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")



            if args.dataset in ['owod'] and epoch % args.eval_every == 0 and epoch > 0:
                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.close()


def get_datasets(args):
    print(args.dataset)
    if args.dataset == 'owod':
        train_set = args.train_set
        test_set = args.test_set
        dataset_train = OWDetection(args, args.owod_path, ["2007"], image_sets=[args.train_set], transforms=make_coco_transforms(args.train_set))
        dataset_val = OWDetection(args, args.owod_path, ["2007"], image_sets=[args.test_set], transforms=make_coco_transforms(args.test_set))
    else:
        raise ValueError("Wrong dataset name")

    print(args.dataset)
    print(args.train_set)
    print(args.test_set)
    print(dataset_train)
    print(dataset_val)

    return dataset_train, dataset_val


def set_dataset_path(args):
    args.owod_path = os.path.join(args.data_root, 'VOC2007')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    set_dataset_path(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # Path(os.path.join(args.pseudo_store_path)).mkdir(parents=True, exist_ok=True)
    main(args)