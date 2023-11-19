#!/usr/bin/env bash
datapath="/data2/yexinyi/datasets/MVS/blendMVS/dataset_low_res/"

resume="<your ckpts>"
log_dir="./checkpoints/DMVSNet/finetune"
if [ ! -d $log_dir ]; then
    mkdir -p $log_dir
fi

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2342 main.py \
        --sync_bn \
        --blendedmvs_finetune \
        --ndepths 48 32 8 \
        --interval_ratio 4 2 1 \
        --img_size 576 768 \
        --dlossw 0.5 1.0 2.0 \
        --log_dir $log_dir \
        --datapath $datapath \
        --resume $resume \
        --dataset_name "blendedmvs" \
        --nviews 7 \
        --epochs 10 \
        --batch_size 1 \
        --lr 0.0001 \
        --scheduler steplr \
        --warmup 0.2 \
        --milestones 6 8 \
        --lr_decay 0.5 \
        --trainlist "datasets/lists/blendedmvs/training_list.txt" \
        --testlist "datasets/lists/blendedmvs/validation_list.txt" \
        --fea_mode "fpn" \
        --agg_mode "variance" \
        --depth_mode "regression" \
        --numdepth 128 \
        --interval_scale 1.06 ${@:1} | tee -a $log_dir/log.txt
