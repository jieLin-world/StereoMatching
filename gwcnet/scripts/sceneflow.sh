#!/usr/bin/env bash
set -x
DATAPATH="/mnt/cephfs/dataset/stereo_matching/sceneflow/"
python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 16 --lrepochs "10,12,14,16:2" \
    --model gwcnet-gc --logdir ./checkpoints/sceneflow/gwcnet-gc