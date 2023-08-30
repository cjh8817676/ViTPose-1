# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
import numpy as np

import json
import pdb
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # pdb.set_trace()
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    
    val_json_path =  cfg.data.test['ann_file']
    
    with open(val_json_path) as json_file:
        data = json.load(json_file)
    
    # pdb.set_trace()
    np_keypoint = np.empty((0, 17, 2))  # 建立空的 NumPy 陣列
    
    for i in data['annotations']:
        temp = []
        temp.append(i['keypoints'][0:2])
        temp.append(i['keypoints'][3:5])
        temp.append(i['keypoints'][6:8])
        temp.append(i['keypoints'][9:11])
        temp.append(i['keypoints'][12:14])
        temp.append(i['keypoints'][15:17])
        temp.append(i['keypoints'][18:20])
        temp.append(i['keypoints'][21:23])
        temp.append(i['keypoints'][24:26])
        temp.append(i['keypoints'][27:29])
        temp.append(i['keypoints'][30:32])
        temp.append(i['keypoints'][33:35])
        temp.append(i['keypoints'][36:38])
        temp.append(i['keypoints'][39:41])
        temp.append(i['keypoints'][42:44])
        temp.append(i['keypoints'][45:47])
        temp.append(i['keypoints'][48:50])
        
        np_keypoint = np.append(np_keypoint, [temp], axis=0)  # 直接將 temp 陣列附加到 np_keypoint 陣列中
    # 如果驗證也是用我自己的資料集, 使用np_keypoint
    # pdb.set_trace()
    np_keypoint = np_keypoint.reshape((-1, 17, 2))  # 重新組成 n x 17 x 2 的形狀


    # pdb.set_trace()
        
    
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority   一般情況下這裡可以註解
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {})
    }

    # pdb.set_trace()
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    # pdb.set_trace()
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[args.gpu_id])
        # pdb.set_trace()
        # outputs = single_gpu_test(model, data_loader)
        outputs, kpt_results = single_gpu_test(model, data_loader, np_keypoint) # 如果驗證也是用我自己的資料集
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    eval_config = cfg.get('evaluation', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))
    coco_pose = ["nose", "leye", "reye", "lear", "rear", "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhip", "rhip", "lknee", "rknee", "lankle", "rankle"]
    
    # 只有在COCO資料集驗證才用到
    # if rank == 0:
    #     if args.out:
    #         print(f'\nwriting results to {args.out}')
    #         mmcv.dump(outputs, args.out)
    #     # pdb.set_trace()
    #     results = dataset.evaluate(outputs, cfg.work_dir, **eval_config)
    #     for k, v in sorted(results.items()):
    #         print(f'{k}: {v}')
    
    
    
    #  只有在客製化資料集驗證才用到
    print('pck kpts accuracy:',kpt_results)
    # 顯示每個關鍵點的準確度
    for k, accuracy in enumerate(kpt_results):
        pose_keypoint = coco_pose[k]
        print(f"Keypoint {pose_keypoint} accuracy: {accuracy}")


if __name__ == '__main__':
    main()
