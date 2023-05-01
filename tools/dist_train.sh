#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# # multi gpu 使用了 python -m torch.distributed.launch
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


# single gpu ->  --launcher none
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG $CHECKPOINT --launcher none ${@:4}



# python -m : 將模塊當做腳本來運行
# 腳本中的import命令，用來引入模塊，引入模塊的過程，也會執行模塊文件暴露在外的代碼。
# 不過，在 if __name__ == '__main__': 下的代碼，不會被執行，因為import時的 __name__不等於__main__，而是當前的模塊名稱。

# 哪什麼時候 __name__ 等於 __main__ 呢？直接運行某個.py文件的時候！
# 要把python的標準庫中的模塊，當成腳本來運行，其實就是運行模塊文件中的包含在 if __name == '__main__': 下的代碼，有些是測試代碼，有些是功能代碼。

# ----------------------------------------------------------------------
# 下面有两个例子，通过不同方式启动同一文件，sys.path属性的值有何不同。
# 文件目录为D:\book\img\run.py，在D:\book为当前目录去执行

# # run.py 内容如下
# import sys
# print(sys.path)

# 直接启动：
# python ./img/run.py
# ['D:\book\img', 'D:\Python\Python38\python38.zip', 'D:\Python\Python38\DLLs', 'D:\Python\Python38\lib', ...]

# 以模块启动
# python -m img.run
# ['D:\book', 'D:\Python\Python38\python38.zip', 'D:\Python\Python38\DLLs', 'D:\Python\Python38\lib',...]

# 细心的同学会发现，在第一行有所不同。
# 直接启动是把run.py文件，所在的目录放到了sys.path属性中。
# 模块启动是把你输入命令的目录（也就是当前路径），放到了sys.path属性中，
# ----------------------------------------------------------------------
# 这个特性有什么用呢??

# 目录结构如下
# p1/
#     __init__.py
#     m.py
# p2/
#     __init__.py
#     run.py
# # run.py 内容如下
# import sys
# from p1 import m
# print(sys.path)

# 如何才能启动run.py文件？

# 直接启动（失败）
# >>>python p2/run.py
# #ImportError: No module named package

# 以模块方式启动（成功）
# >>>python -m p2.run
# 当需要启动的py文件引用了一个模块。你需要注意：在启动的时候需要考虑sys.path中有没有你import的模块的路径！
# 这个时候，到底是使用直接启动，还是以模块的启动？目的就是把import的那个模块的路径放到sys.path中。
