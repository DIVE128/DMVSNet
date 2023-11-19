#!/usr/bin/env bash
datapath="/data2/yexinyi/datasets/MVS/preprocessed_inputs/tankandtemples/advanced/"
# datapath="/data2/yexinyi/datasets/MVS/preprocessed_inputs/tankandtemples/intermediate/"

outdir="./outputs_tank/DMVSNet/"
resume="<your ckpts>"

CUDA_VISIBLE_DEVICES=1 python main.py \
        --test \
        --ndepths 64 32 8 \
        --interval_ratio 3 2 1 \
        --num_view 11 \
        --outdir $outdir \
        --datapath $datapath \
        --resume $resume \
        --dataset_name "general_eval" \
        --batch_size 1 \
        --testlist "all" \
        --fea_mode "fpn" \
        --agg_mode "variance" \
        --depth_mode "regression" \
        --numdepth 192 \
        --interval_scale 1.06 \
        --filter_method "dypcd" ${@:1}