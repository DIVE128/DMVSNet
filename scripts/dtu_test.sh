#!/usr/bin/env bash
datapath="/data2/yexinyi/datasets/MVS/preprocessed_inputs/dtu/"
outdir="./outputs_dtu/DMVSNet/"
resume="./checkpoints/DMVSNet/model.ckpt"
fusibile_exe_path="./fusibile/build/fusibile"


CUDA_VISIBLE_DEVICES=7 python main.py \
        --test \
        --ndepths 48 32 8 \
        --interval_ratio 4 2 1 \
        --max_h 864 \
        --max_w 1152 \
        --num_view 5 \
        --outdir $outdir \
        --datapath $datapath \
        --resume $resume \
        --dataset_name "general_eval" \
        --batch_size 1 \
        --testlist "datasets/lists/dtu/test.txt" \
        --fea_mode "fpn" \
        --agg_mode "variance" \
        --depth_mode "regression" \
        --numdepth 192 \
        --interval_scale 1.06 \
        --filter_method "pcd" \
        --thres_view 5 \
        --num_worker 1 \
        --inverse_depth \
        --conf 0. 0. 0.3 ${@:1}
