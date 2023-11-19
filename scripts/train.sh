#!/usr/bin/env bash
datapath="/data2/yexinyi/datasets/MVS/training_data/dtu_training/"

log_dir="checkpoints/DMVSNet"
if [ ! -d $log_dir ]; then
    mkdir -p $log_dir
fi

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1111 main.py \
        --sync_bn \
        --ndepths 48 32 8 \
        --interval_ratio 4 2 1 \
        --img_size 512 640 \
        --num_view 5 \
        --dlossw 0.5 1.0 2.0 \
        --log_dir $log_dir \
        --datapath $datapath \
        --dataset_name "dtu_yao" \
        --epochs 16 \
        --batch_size 2 \
        --lr 0.001 \
        --warmup 0.2 \
        --scheduler "steplr" \
        --milestones 10 12 14 \
        --lr_decay 0.5 \
        --trainlist "datasets/lists/dtu/train.txt" \
        --testlist "datasets/lists/dtu/test.txt" \
        --fea_mode "fpn" \
        --agg_mode "variance" \
        --depth_mode "regression" \
        --inverse_depth \
        --numdepth 192 \
        --interval_scale 1.06 ${@:1} | tee -a $log_dir/log.txt

