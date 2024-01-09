#!/usr/bin/env bashtest

set -x

EXP_DIR=exps_lvisnococo/OWOD_t1_new_split
PY_ARGS=${@:1}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --train_set 't1_train_owdetr' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 45 --top_unk 5 --featdim 1024 --NC_branch  --nc_epoch 5 \
    --backbone 'dino_resnet50' --objectness 'object_score' --pseudo --use_tensorboard \
    --resume '/mnt/gluster/home/mashuailei/DOWB/exps_lvisnococo/OWOD_t1_new_split/checkpoint.pth' \
    ${PY_ARGS}

# EXP_DIR=exps_new_split/OWOD_t2_new_split
# PY_ARGS=${@:1}

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
#     --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
#     --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --train_set 't2_train_owdetr' --test_set 'test' --num_classes 81 \
#     --unmatched_boxes --epochs 50 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch  --nc_epoch 5 \
#     --backbone 'dino_resnet50' --objectness 'object_score' --pseudo  --use_tensorboard \
#     --pretrain 'exps_new_split/OWOD_t1_new_split/checkpoint0044.pth' \
#     ${PY_ARGS}

# EXP_DIR=exps_new_split/OWOD_t2_ft_new_split
# PY_ARGS=${@:1}

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
#     --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
#     --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --train_set 't2_ft_owdetr' --test_set 'test' --num_classes 81 \
#     --unmatched_boxes --epochs 100 --top_unk 5 --featdim 1024 --NC_branch  --nc_epoch 5 \
#     --backbone 'dino_resnet50' --objectness 'object_score' --pseudo --use_tensorboard \
#     --pretrain 'exps_new_split/OWOD_t2_new_split/checkpoint0049.pth' \
#     ${PY_ARGS}

# EXP_DIR=exps_new_split/OWOD_t3_new_split
# PY_ARGS=${@:1}

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
#     --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
#     --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --train_set 't3_train_owdetr' --test_set 'test' --num_classes 81 \
#     --unmatched_boxes --epochs 106 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch  --nc_epoch 5 \
#     --backbone 'dino_resnet50' --objectness 'object_score' --pseudo --use_tensorboard \
#     --pretrain 'exps_new_split/OWOD_t2_ft_new_split/checkpoint0099.pth' \
#     ${PY_ARGS}

# EXP_DIR=exps_new_split/OWOD_t3_ft_new_split
# PY_ARGS=${@:1}

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
#     --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
#     --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --train_set 't3_ft_owdetr' --test_set 'test' --num_classes 81 \
#     --unmatched_boxes --epochs 161 --top_unk 5 --featdim 1024 --NC_branch  --nc_epoch 5 \
#     --backbone 'dino_resnet50' --objectness 'object_score' --pseudo --use_tensorboard \
#     --pretrain 'exps_new_split/OWOD_t3_new_split/checkpoint0104.pth' \
#     ${PY_ARGS}

# EXP_DIR=exps_new_split/OWOD_t4_new_split
# PY_ARGS=${@:1}

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
#     --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
#     --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --train_set 't4_train_owdetr' --test_set 'test' --num_classes 81 \
#     --unmatched_boxes --epochs 171 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch  --nc_epoch 5 \
#     --backbone 'dino_resnet50' --objectness 'object_score' --pseudo --use_tensorboard \
#     --pretrain 'exps_new_split/OWOD_t3_ft_new_split/checkpoint0159.pth' \
#     ${PY_ARGS}

# EXP_DIR=exps_new_split/OWOD_t4_ft_new_split
# PY_ARGS=${@:1}

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
#     --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
#     --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --train_set 't4_ft_owdetr' --test_set 'test' --num_classes 81 \
#     --unmatched_boxes --epochs 302 --top_unk 5 --featdim 1024 --NC_branch  --nc_epoch 5 \
#     --backbone 'dino_resnet50' --objectness 'object_score' --pseudo --use_tensorboard \
#     --pretrain 'exps_new_split/OWOD_t4_new_split/checkpoint0169.pth' \
#     ${PY_ARGS}