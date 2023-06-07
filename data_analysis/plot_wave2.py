import os
import json
import pandas as pd

'''
l_eye.append(i['keypoints'][3:5])
r_eye.append(i['keypoints'][6:8])
l_ear.append(i['keypoints'][9:11])
r_ear.append(i['keypoints'][12:14])
l_shoulder.append(i['keypoints'][15:17])
r_shoulder.append(i['keypoints'][18:20])
l_elbow.append(i['keypoints'][21:23])
r_elbow.append(i['keypoints'][24:26])
l_wrist.append(i['keypoints'][27:29])
r_wrist.append(i['keypoints'][30:32])
l_hip.append(i['keypoints'][33:35])
r_hip.append(i['keypoints'][36:38])
l_knee.append(i['keypoints'][39:41])
r_knee.append(i['keypoints'][42:44])
l_ankle.append(i['keypoints'][45:47])
r_ankle.append(i['keypoints'][48:50])
'''

def read_files(sub_folder):
    base_path = "output_results"
    sub_path = os.path.join(base_path, sub_folder)
    files = os.listdir(sub_path)

    fail_files = []
    success_files = []

    for file in files:
        if file.endswith('.json'):
            if "fail" in file:
                fail_files.append(os.path.join(sub_path, file))
            elif "success" in file:
                success_files.append(os.path.join(sub_path, file))

    return fail_files, success_files


if __name__ == "__main__":
    
    # 讀取 'sub1' 的所有檔案
    fail_files, success_files = read_files('sub3')
    
    fail_json = []
    succcess_json = []
    
    # read results of pose estimation.
    for file in success_files:
        with open(file) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
            succcess_json.append(json.load(f))
    for file in fail_files:
        with open(file) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
            fail_json.append(json.load(f))
            
    success_hip_angle_data = []
    fail_hip_angle_data = []
    success_shoulder_angle_data = []
    fail_shoulder_angle_data = []
    
    # make json becaome pandas:
    for suc in succcess_json:
        for data in suc:
            print(data['keypoints'])