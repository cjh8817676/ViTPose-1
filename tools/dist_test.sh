#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=$1               # bash 指令的第一個參數
CHECKPOINT=$2           # bash 指令的第一個參數
GPUS=$3
PORT=${PORT:-29500}

# # multi gpu 使用了 python -m torch.distributed.launch
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}


# single gpu => --launcher none ,   ${@:4} : 執行4次
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher none ${@:4}



# 參考資料:
# https://zhuanlan.zhihu.com/p/384893917
