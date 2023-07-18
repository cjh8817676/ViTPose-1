import os
import json
import numpy as np
import pandas as pd
import pdb
import math
import matplotlib.pyplot as plt
import re
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

'''
read .json(predict) and .xlsx(gt)  and compare their difference and visualization.
'''


def moving_average_filter(data, window_size = 5):
    filtered_data = []

    for i in range(window_size, len(data) - window_size):
        window = data[i - window_size: i + window_size + 1]
        avg = sum(window) / len(window)
        filtered_data.append(avg)

    # 補償損失的資料點
    compensate_points = window_size // 2
    compensated_data = data[:compensate_points] + filtered_data + data[-compensate_points:]

    return compensated_data

def median_filter(signal, window_size):
    """
    對一維訊號做中值濾波。
    :param signal: 一維訊號
    :param window_size: 窗口大小
    :return: 濾波後的訊號
    """
    filtered_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        lower_bound = max(0, i - window_size // 2)
        upper_bound = min(len(signal), i + window_size // 2 + 1)
        filtered_signal[i] = np.median(signal[lower_bound:upper_bound])
    return filtered_signal
def calculate_angle_left_body(pointlist):

    p1, p2, p3 = pointlist[0],pointlist[1],pointlist[2]
    """
    Calculates the angle between three points in 2D space.
    Args:
        p1, p2, p3: Each point is a list [x, y]
    Returns:
        angle: angle in degrees
    """
    temp = p1
    p1 = p2
    p2 = temp
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1, 1))
    
    # Adjust the angle's range to [90, 270] degrees if needed.
    if np.cross(v1, v2) > 0:
        angle = 2*np.pi - angle
    
    return np.degrees(angle)


def calculate_angle_right_body(pointlist):
    """
    input a list which have 3 points
    [(hip),(shoulder),(knee)]
    The maximum bending angle that a person can achieve is 0~270 degrees5
    """
    p1 = pointlist[0]
    p2 = pointlist[1]
    p3 = pointlist[2]
    temp = p1
    p1 = p2
    p2 = temp
    # Calculate the angle between the vectors (p1, p2) and (p3, p2)
    angle = math.atan2(p3[1]-p2[1], p3[0]-p2[0]) - math.atan2(p1[1]-p2[1], p1[0]-p2[0])
    angle = math.degrees(angle)
    
    return angle+360 if angle < 0 else angle

def read_files(sub_folder,action):
    base_path = "output_results"
    sub_path = os.path.join(base_path, sub_folder)
    files = os.listdir(sub_path)

    fail_files = []
    success_files = []

    for file in files:
        if action in file:
            if file.endswith('.json'):
                if "fail" in file:
                    fail_files.append(os.path.join(sub_path, file))
                elif "success" in file:
                    success_files.append(os.path.join(sub_path, file))

    return fail_files, success_files

def read_gtfiles(sub,action):
    base_path = "tracker_gt_output_results"
    files = os.listdir(base_path)

    gt_fail_files = []
    gt_success_files = []
    for file in files:
        if action in file:
            if sub in file and 'joint' in file:
                if "fail" in file:
                    gt_fail_files.append(os.path.join(base_path,file))
                elif "success" in file:
                    gt_success_files.append(os.path.join(base_path,file))
               
    return gt_fail_files,gt_success_files

def process_json_data(json_data):
    keypoints = ['l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
                 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    body_parts = {
        'l_eye': slice(3, 5),    # pdb.set_trace()
    # fail_files = sorted(fail_files)
    # success_files = sorted(success_files)
        'r_eye': slice(6, 8),
        'l_ear': slice(9, 11),
        'r_ear': slice(12, 14),
        'l_shoulder': slice(15, 17),
        'r_shoulder': slice(18, 20),
        'l_elbow': slice(21, 23),
        'r_elbow': slice(24, 26),
        'l_wrist': slice(27, 29),
        'r_wrist': slice(30, 32),
        'l_hip': slice(33, 35),
        'r_hip': slice(36, 38),
        'l_knee': slice(39, 41),
        'r_knee': slice(42, 44),
        'l_ankle': slice(45, 47),
        'r_ankle': slice(48, 50)
    }

    keypoints_data = {part: [] for part in keypoints}
    angle_data = {'left_hip': [], 'left_shoulder': [], 'right_hip': [], 'right_shoulder': []}

    for i in json_data:
        for part, indices in body_parts.items():
            keypoints_data[part].append(i['keypoints'][indices])

    for counter in range(len(json_data)):
        angle_data['left_hip'].append(calculate_angle_left_body([keypoints_data['l_hip'][counter],
                                                                 keypoints_data['l_knee'][counter],
                                                                 keypoints_data['l_shoulder'][counter]]))
        angle_data['left_shoulder'].append(calculate_angle_left_body([keypoints_data['l_shoulder'][counter],
                                                                      keypoints_data['l_wrist'][counter],
                                                                      keypoints_data['l_hip'][counter]]))
        angle_data['right_hip'].append(calculate_angle_right_body([keypoints_data['r_hip'][counter],
                                                                   keypoints_data['r_knee'][counter],
                                                                   keypoints_data['r_shoulder'][counter]]))
        angle_data['right_shoulder'].append(calculate_angle_right_body([keypoints_data['r_shoulder'][counter],
                                                                        keypoints_data['r_wrist'][counter],
                                                                        keypoints_data['r_hip'][counter]]))

    return angle_data



if __name__ == "__main__":
    # pdb.set_trace()
    # 讀取 'subn' 的所有檔案, 並指定動作: YAMAKI, TKATCHEV
    subject = 'sub4'
    action = 'TKATCHEV'
    
    fail_files, success_files = read_files(subject,action)
    
    gt_fail_files,gt_success_files = read_gtfiles(subject,action)

    
    fail_files = sorted(fail_files, key=lambda x: int(re.search(f'{subject}_(\d+)_', x).group(1)))
    success_files = sorted(success_files, key=lambda x: int(re.search(f'{subject}_(\d+)_', x).group(1)))
    gt_fail_files = sorted(gt_fail_files, key=lambda x: int(re.search(f'{subject}_(\d+)_', x).group(1)))
    gt_success_files = sorted(gt_success_files, key=lambda x: int(re.search(f'{subject}_(\d+)_', x).group(1)))

    # pdb.set_trace()
    '''
    read gt file
    '''
    fail_xlsx = []
    success_xlsx = []
    
    for file in gt_success_files:
        if action in file:
            pd_data = pd.read_excel(file)
            pd_data = pd_data.iloc[1:]
            pd_data.rename(columns = {f'{pd_data.columns[0]}': 'time'}, inplace = True)
            pd_data.rename(columns = {f'{pd_data.columns[1]}': 'step'}, inplace = True)
            pd_data.rename(columns = {'Unnamed: 3': 'shoulder_ω'}, inplace = True)
            pd_data.rename(columns = {'Unnamed: 5': 'hip_ω'}, inplace = True)
            success_xlsx.append(pd_data)
            
    for file in gt_fail_files:
        if action in file:
            # pdb.set_trace()
            pd_data = pd.read_excel(file)
            pd_data = pd_data.iloc[1:]
            pd_data.rename(columns = {f'{pd_data.columns[0]}': 'time'}, inplace = True)
            pd_data.rename(columns = {f'{pd_data.columns[1]}': 'step'}, inplace = True)
            pd_data.rename(columns = {'Unnamed: 3': 'shoulder_ω'}, inplace = True)
            pd_data.rename(columns = {'Unnamed: 5': 'hip_ω'}, inplace = True)
            fail_xlsx.append(pd_data)
            
    # pdb.set_trace()
    success_hip_angle_data = []
    fail_hip_angle_data = []
    success_shoulder_angle_data = []
    fail_shoulder_angle_data = []
    
    for i in fail_xlsx:
        fail_shoulder_angle_data.append(i['shoulder_angle'].tolist())
        fail_hip_angle_data.append(i['hip_angle'].tolist())
    counter = 0
    for i in success_xlsx:
        success_shoulder_angle_data.append(i['shoulder_angle'].tolist())
        success_hip_angle_data.append(i['hip_angle'].tolist())
        print('counter',counter)
        counter += 1

    # pdb.set_trace()
    
    '''
    read predict file
    '''
    fail_json = []            
    success_json = []
    
    # read results of pose estimation.
    for file in success_files:
        if action in file:
            print(file)
            with open(file) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
                success_json.append(json.load(f))
    for file in fail_files:
        if action in file:
            print(file)
            with open(file) as f:    # 讀取每一幀的 pose keypoint 和 bbox(左上、右下) 的座標
                fail_json.append(json.load(f))
            
            
    # pdb.set_trace()
    
    success_rhip_angle_data = []
    fail_rhip_angle_data = []
    success_lhip_angle_data = []
    fail_lhip_angle_data = []
    
    success_rshoulder_angle_data = []
    fail_rshoulder_angle_data = []
    success_lshoulder_angle_data = []
    fail_lshoulder_angle_data = []
    
    for json_data in success_json:
        angle_data = process_json_data(json_data)
        success_rhip_angle_data.append(angle_data['right_hip'])
        success_lhip_angle_data.append(angle_data['left_hip'])
        success_rshoulder_angle_data.append(angle_data['right_shoulder'])
        success_lshoulder_angle_data.append(angle_data['left_shoulder'])
    
    for json_data in fail_json:
        angle_data = process_json_data(json_data)
        fail_rhip_angle_data.append(angle_data['right_hip'])
        fail_lhip_angle_data.append(angle_data['left_hip'])
        fail_rshoulder_angle_data.append(angle_data['right_shoulder'])
        fail_lshoulder_angle_data.append(angle_data['left_shoulder'])   
    # 将所有数据列表放在一个列表中 (according to video to choose left or right )
    
    # data_lists = [
        # fail_rhip_angle_data,
        # fail_lhip_angle_data,
        # fail_rshoulder_angle_data,
        # fail_lshoulder_angle_data,
        # success_rhip_angle_data,
        # success_lhip_angle_data,
        # success_rshoulder_angle_data
        # success_lshoulder_angle_data
    # ]
    
    
    # sub1 : left hip,all right
    if subject=='sub1':
        # all YAMAWAKI
        data_lists = [
            fail_lhip_angle_data,
            fail_rshoulder_angle_data,
            success_lhip_angle_data,
            success_rshoulder_angle_data
        ]
        # 有個地方要注意的是，每有指定左('l')右('r')的指的就是ground truth。
        gt_data_lists = [fail_hip_angle_data,
                         fail_shoulder_angle_data,
                         success_hip_angle_data,
                         success_shoulder_angle_data]

     # sub2 : left hip, rigth shoulder
    elif subject=='sub2':
        if action == 'YAMAWAKI':
            # pdb.set_trace()
            # 前3次後3次換邊執行運動了
            success_lhip_angle_data[-3:] = success_rhip_angle_data[-3:]
            success_rshoulder_angle_data[-3:] = success_lshoulder_angle_data[-3:]
            # all success in YAMAWAKI
            data_lists = [
                success_lhip_angle_data,      #  這裡確實有問題，有待改善
                success_rshoulder_angle_data
            ]
            gt_data_lists = [
                success_hip_angle_data,
                success_shoulder_angle_data
                ]
        
        elif action =='TKATCHEV':
            # 2 fail 1 success in TKATCHEV
            data_lists = [
                fail_lhip_angle_data,
                fail_rshoulder_angle_data,
                success_lhip_angle_data,
                success_rshoulder_angle_data
            ]
            gt_data_lists = [
                fail_hip_angle_data,
                fail_shoulder_angle_data,
                success_hip_angle_data,
                success_shoulder_angle_data
                ]
    elif subject== 'sub3':
    # all TKATCHEV
    # sub3 : all left
        data_lists = [
            fail_rhip_angle_data,
            fail_lshoulder_angle_data,
            success_rhip_angle_data,
            success_lshoulder_angle_data
        ]
        gt_data_lists = [
            fail_hip_angle_data,
            fail_shoulder_angle_data,
            success_hip_angle_data,
            success_shoulder_angle_data
            ]
    
    elif subject== 'sub4':
        if action == 'YAMAWAKI':
            # pdb.set_trace()
            # all success in YAMAWAKI
            data_lists = [
                success_lhip_angle_data,      #  這裡確實有問題，有待改善
                success_rshoulder_angle_data,
                success_lhip_angle_data,      #  這裡確實有問題，有待改善
                success_rshoulder_angle_data
            ]
            gt_data_lists = [
                fail_hip_angle_data,
                fail_shoulder_angle_data,
                success_hip_angle_data,
                success_shoulder_angle_data
                ]
        
        elif action =='TKATCHEV':
            # ALL success in TKATCHEV
            data_lists = [
                success_rhip_angle_data,
                success_lshoulder_angle_data
            ]
            gt_data_lists = [
                success_hip_angle_data,
                success_shoulder_angle_data
                ]
        
    
    # pdb.set_trace()
    # 设置图形大小plt.figure(figsize=(10, 6)) 
    
    # plt.figure(figsize=(10, 6)) 
    
    # selecct sub1,sub2,sub3
    if subject=='sub1':
        
        # all YAMAWAKI
        sub1_list = [[(0,410)],[(0,410)],[(0,410),(0,426),(0,440),(0,480),(0,412),(0,449)],[(0,410),(0,426),(0,440),(0,480),(0,412),(0,449)]] # for fail and success
        # sub1_list = [[(0,410),(0,426),(0,440)],[(0,410),(0,426),(0,440)]] # for success
        # sub1_list = [[(0,410)],[(0,410)]] # for fail
        results = [['fail'],['fail'],['success','success','success','success','success','success'],['success','success','success','success','success','success']]
        orders = [[1],[1],[2,3,4,5,6,7],[2,3,4,5,6,7]]
        sub_list = sub1_list
        color = [['r'],['b'],['r','r','r','r','r','r'],['b','b','b','b','b','b']]
        gt_color = [['g'],['m'],['g','g','g','g','g','g'],['m','m','m','m','m','m']]
        
    elif subject=='sub2':
        
        if action=='YAMAWAKI':
            sub2_list = [[(0,152),(0,163),(0,159),(0,178),(0,170),(0,178)],[(0,152),(0,163),(0,159),(0,178),(0,170),(0,178)]] # for succss
            results = [['success','success','success','success','success','success'],['success','success','success','success','success','success']]
            orders = [[1,2,3,4,5,6],[1,2,3,4,5,6]]
            sub_list = sub2_list
            color = [['r','r','r','r','r','r'],['b','b','b','b','b','b']]
            gt_color = [['g','g','g','g','g','g'],['m','m','m','m','m','m']]
        elif action == 'TKATCHEV':
            sub2_list = [[(0,404),(0,425)],[(0,404),(0,425)],[(0,425)],[(0,425)]]
            results = [['fail','fail'],['fail','fail'],['success'],['success']]
            orders = [[1,3],[1,3],[2],[2]]
            sub_list = sub2_list
            color = [['r','r'],['b','b'],['r','r'],['b','b']]
            gt_color = [['g','g'],['m','m'],['g','g'],['m','m']]
            
    elif subject== 'sub3':
        # sub3_list are set by observing the frame of film from hand stand to release.
        # [fail;fail;success;success]
        sub3_list = [[(300,735),(314,740),(255,690),(0,429)],[(300,735),(314,740),(255,690),(0,429)],[(263,715),(296,740),(303,751),(0,461),(0,450),(0,441),(0,445),(0,433)],[(263,715),(296,740),(303,751),(0,461),(0,450),(0,441),(0,445),(0,433)]] # for fail and success
        # sub3_list = [[(0,715),(0,740),(0,751)],[(0,715),(0,740),(0,751)]] # for succss
        results = [['fail','fail','fail','fail'],['fail','fail','fail','fail'],['success','success','success','success','success','success','success','success'],['success','success','success','success','success','success','success','success']]
        sub_list = sub3_list
        orders = [[1,2,3,11],[1,2,3,11],[4,5,6,7,8,9,10,12],[4,5,6,7,8,9,10,12]]
        color = [['r','r','r','r'],['b','b','b','b'],['r','r','r','r','r','r','r','r'],['b','b','b','b','b','b','b','b']]
        gt_color = [['g','g','g','g'],['m','m','m','m'],['g','g','g','g','g','g','g','g'],['m','m','m','m','m','m','m','m']]
    
    elif subject== 'sub4':
        if action=='YAMAWAKI':
            # pdb.set_trace()
            # [fail;fail;success;success]
            sub4_list = [[(0,447)],[(0,447)],[(0,425),(0,437)],[(0,425),(0,437)]] # for fail and success
            # sub3_list = [[(0,715),(0,740),(0,751)],[(0,715),(0,740),(0,751)]] # for succss
            results = [['fail'],['fail'],['success','success'],['success','success']]
            sub_list = sub4_list
            orders = [[3],[3],[1,2],[1,2]]
            color = [['r'],['b'],['r','r'],['b','b']]
            gt_color = [['g'],['m'],['g','g'],['m','m']]
        if action =='TKATCHEV':
            # all succss in TKATCHEV
            sub4_list = [[(0,506),(0,508),(0,474)],[(0,506),(0,508),(0,474)]] # for fail and success
            results = [['success','success','success'],['success','success','success']]
            sub_list = sub4_list
            orders = [[1,2,3],[1,2,3]]
            color = [['r','r','r'],['b','b','b',]]
            gt_color = [['g','g','g'],['m','m','m']]

    
    
    min_value = min(min(sub_list, key=min))
    min_value = min_value[1]
    
    plt.plot([], color='b', label='Shoulder Angle')
    plt.plot([], color='r', label='Hip Angle')
 
    
    mae_list = []  
    rmse_list = []  
    std_list = []
    # pdb.set_trace()
    count = 0
    # 循环遍历数据列表并绘制波形图
    for data_list,gt_data_list,margin,cols,gt_cols,order,result in zip(data_lists,gt_data_lists,sub_list,color,gt_color,orders,results):
        for j,k,marg,col,gt_col,ind,res in zip(data_list,gt_data_list,margin,cols,gt_cols,order,result):
            plt.figure(figsize=(14, 10)) 
            # pdb.set_trace()
            
            temp = j[marg[0]:marg[1]]
            
            temp = temp[::-1]
            
            temp = temp[0:min_value]
            
            j = temp[::-1]
            
            
            threshold = 40
            diff = np.abs(np.diff(temp))
            if np.any(diff > threshold):
                print('median')
                j = median_filter(j,6)  # predict joint angle pass moving avrage filter
                k = median_filter(k,6)  # tracker gt joint angle pass moving avrage filter
                # j = moving_average_filter(j.tolist(),6)  # predict joint angle pass moving avrage filter
                # k = moving_average_filter(k.tolist(),6)  # tracker gt joint angle pass moving avrage filter
                # j = median_filter(j, threshold)
            else:
                print('moving')
                # j = moving_average_filter(j,5)
                j = moving_average_filter(j,6)  # predict joint angle pass moving avrage filter
                k = moving_average_filter(k,6)  # tracker gt joint angle pass moving avrage filter
            
            
            
            count += 1
            
            # 生成 x 轴坐标，从 0 到数据列表长度减 1
            x = [i/120 for i in range(len(j))]
            # 绘制波形图
            k = k[::-1]
            k = k[0:len(j)]
            k = k[::-1]
            
            diff = np.subtract(np.array(j), np.array(k))
            
            # std
            std = np.std(diff)
            std_list.append(std)
            # mae
            abs_diff = np.abs(diff)
            mae = np.mean(abs_diff)  # mae of predict and gt.
            mae_list.append(mae)
            # rmse 
            squared_diff = np.square(diff) 
            mse = np.mean(squared_diff)
            rmse = np.sqrt(mse)
            rmse_list.append(rmse)
            
            if col == 'r':
                body_part = "hip"
                plt.plot(x, j, color=col, label='Predict')
            elif col == 'b':
                body_part = "shoulder"
                plt.plot(x, j, color=col, label='Predict')
    
            if gt_col == 'g':
                body_part = "hip"
                plt.plot(x, k, color=gt_col, label='GT')
            elif gt_col == 'm':
                body_part = "shoulder"
                plt.plot(x, k, color=gt_col, label='GT')

            # 根据颜色设置图例标签
            legend  = plt.legend(loc='lower left')
            for line in legend.get_lines():
                line.set_linewidth(10) 
            for text in legend.get_texts():
                text.set_fontsize(35)  # 设置字体大小
            plt.xlabel('time(s)',fontsize=70)
            plt.ylabel('Angle',fontsize=70)
            plt.xticks(fontsize=40)
            plt.yticks([120,150,180,210,230], ['120°','150°','180°','210°','230°'],fontsize=40)
            plt.ylim(110, 240)
            # 设置 x 轴和 y 轴刻度标签的数量
            plt.locator_params(axis='x', nbins=5)
            plt.locator_params(axis='y', nbins=8)
            plt.grid(True, linewidth=0.8)
            plt.tight_layout()
            # 保存图形
            plt.savefig("/home/m11002125/" + f'vis_{subject}_{ind}_{action}_{res}_{body_part}')

            
            plt.show()
    
    

    final_std = np.mean(std_list)
    final_rmse = np.mean(rmse_list)
    final_mae = np.mean(mae_list)
    
    print('final_std',final_std)
    print('final_rmse',final_rmse)
    print('final_mae',final_mae)
        
    

            
            
            
            
            