"""
A simple user interface for XMem
"""

import os
# fix for Windows
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
import pdb
import sys
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


import torch
file_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_path,'../XMem-1'))
from model.network import XMem
from inference.interact.s2m_controller import S2MController
from inference.interact.fbrs_controller import FBRSController
from inference.interact.s2m.s2m_network import deeplabv3plus_resnet50 as S2M

from PyQt5.QtWidgets import QApplication
from inference.interact.gui_mmpose import App
from inference.interact.resource_manager import ResourceManager

torch.set_grad_enabled(False)

# Arguments parsing
parser = ArgumentParser()


'''----------------------------vitpose doption-----------------------------'''
parser.add_argument('--det_config', help='Config file for detection')
parser.add_argument('--det_checkpoint', help='Checkpoint file for detection')
parser.add_argument('--pose_config', help='Config file for pose')
parser.add_argument('--pose_checkpoint', help='Checkpoint file for pose')
parser.add_argument('--video-path', type=str, help='Video path')
parser.add_argument('--show',action='store_true',default=False,help='whether to show visualizations.')
parser.add_argument('--out-video-root',default='',help='Root of the output video file. Default not saving the visualization video.')
parser.add_argument('--device', default='cuda:0', help='Device used for inference')
parser.add_argument('--det-cat-id',type=int,default=1,help='Category id for bounding box detection model')
parser.add_argument('--bbox-thr',type=float,default=0.3,help='Bounding box score threshold')
parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
parser.add_argument('--radius',type=int,default=4,help='Keypoint radius for visualization')
parser.add_argument('--thickness',type=int,default=1, help='Link thickness for visualization')

'''----------------------------xmem doption-----------------------------'''
parser.add_argument('--model', default='XMem-1/saves/XMem.pth')
parser.add_argument('--s2m_model', default='XMem-1/saves/s2m.pth')
parser.add_argument('--fbrs_model', default='XMem-1/saves/fbrs.pth')

"""
Priority 1: If a "images" folder exists in the workspace, we will read from that directory
Priority 2: If --images is specified, we will copy/resize those images to the workspace
Priority 3: If --video is specified, we will extract the frames to the workspace (in an "images" folder) and read from there

In any case, if a "masks" folder exists in the workspace, we will use that to initialize the mask
That way, you can continue annotation from an interrupted run as long as the same workspace is used.
"""
parser.add_argument('--images', help='Folders containing input images.', default=None)
parser.add_argument('--video', help='Video file readable by OpenCV.', default=None)
parser.add_argument('--workspace', help='directory for storing buffered images (if needed) and output masks', default=None)

parser.add_argument('--buffer_size', help='Correlate with CPU memory consumption', type=int, default=100)

parser.add_argument('--num_objects', type=int, default=1)

# Long-memory options
# Defaults. Some can be changed in the GUI.
parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128) 

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', type=int, default=10)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)
parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
parser.add_argument('--size', default=480, type=int, 
        help='Resize the shorter side to this size. -1 to use original resolution. ')

args = parser.parse_args()
# pdb.set_trace()
config = vars(args)
config['enable_long_term'] = True
config['enable_long_term_count_usage'] = True

config['video'] = args.video_path
xmem_config = config  # 可以藉由 xmem_config['device'] 取得資料
pose_config = args    # 可以藉由 pose_config.device 取得資料

if __name__ == '__main__':
    with torch.cuda.amp.autocast(enabled=not args.no_amp):

        # Load our checkpoint
        network = XMem(config, args.model).cuda().eval()

        # Loads the S2M model
        if args.s2m_model is not None:
            s2m_saved = torch.load(args.s2m_model)
            s2m_model = S2M().cuda().eval()
            s2m_model.load_state_dict(s2m_saved)
        else:
            s2m_model = None

        s2m_controller = S2MController(s2m_model, args.num_objects, ignore_class=255)
        if args.fbrs_model is not None:
            fbrs_controller = FBRSController(args.fbrs_model)
        else:
            fbrs_controller = None

        # Manages most IO
        resource_manager = ResourceManager(config)

        app = QApplication(sys.argv)
        ex = App(network, resource_manager, s2m_controller, fbrs_controller, xmem_config, pose_config)
        sys.exit(app.exec_())
