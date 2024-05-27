#!/usr/bin/env bash
set -x
DATAPATH="/mnt/cephfs/dataset/stereo_matching/kitti2015"
python main.py \
    --model gwcnet-gc \
    --maxdisp 192 \
    --dataset kitti \
    --datapath $DATAPATH \
    --basepath /mnt/cephfs/dataset/stereo_matching/output/gwcnet/ \
    --textpath kitti15_text.txt \
    --trainlist /mnt/cephfs/home/zhihongyan/linjie/stereo/gwcnet/filenames/kitti15_train.txt \
    --testlist /mnt/cephfs/home/zhihongyan/linjie/stereo/gwcnet/filenames/kitti15_val.txt \
    --lr 0.001 \
    --batch_size 16 \
    --test_batch_size 16 \
    --epochs 300 \
    --lrepochs "200:10" \
    --logdir /mnt/cephfs/dataset/stereo_matching/output/gwcnet/checkpoints/kitti15/gwcnet-gc \
    --loadckpt 
    --test_batch_size 1