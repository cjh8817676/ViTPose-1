# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

def moving_average_filter(data, window_size = 5):
    filtered_data = []
    data = data.to_list()
    for i in range(window_size, len(data) - window_size):
        window = data[i - window_size: i + window_size + 1]
        avg = sum(window) / len(window)
        filtered_data.append(avg)

    # 補償損失的資料點
    # compensate_points = window_size // 2
    compensated_data = data[:window_size] + filtered_data + data[-window_size:]

    return compensated_data

# sub1_filepaths = [
#     'tracker_gt_output_results/vis_sub1_1_YAMAWAKI_fail_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub1_2_YAMAWAKI_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub1_3_YAMAWAKI_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub1_4_YAMAWAKI_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub1_5_YAMAWAKI_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub1_6_YAMAWAKI_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub1_7_YAMAWAKI_success_joint_gt.xlsx',
# ]
# sub2_files 
# sub1_filepaths = [
    # 'tracker_gt_output_results/vis_sub2_1_YAMAWAKI_success_joint_gt.xlsx',
    # 'tracker_gt_output_results/vis_sub2_2_YAMAWAKI_success_joint_gt.xlsx',
    # 'tracker_gt_output_results/vis_sub2_3_YAMAWAKI_success_joint_gt.xlsx',
    # 'tracker_gt_output_results/vis_sub2_4_YAMAWAKI_success_joint_gt.xlsx',
    # 'tracker_gt_output_results/vis_sub2_5_YAMAWAKI_success_joint_gt.xlsx',
    # 'tracker_gt_output_results/vis_sub2_6_YAMAWAKI_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub2_1_TKATCHEV_fail_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub2_2_TKATCHEV_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub2_3_TKATCHEV_fail_joint_gt.xlsx',
# ]
# sub3_files
# sub1_filepaths = [
#     'tracker_gt_output_results/vis_sub3_1_TKATCHEV_fail_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_2_TKATCHEV_fail_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_3_TKATCHEV_fail_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_4_TKATCHEV_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_5_TKATCHEV_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_6_TKATCHEV_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_7_TKATCHEV_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_8_TKATCHEV_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_9_TKATCHEV_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_10_TKATCHEV_success_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_11_TKATCHEV_fail_joint_gt.xlsx',
#     'tracker_gt_output_results/vis_sub3_12_TKATCHEV_success_joint_gt.xlsx'
# ]

# sub4_files
sub1_filepaths = [
    'tracker_gt_output_results/vis_sub4_1_YAMAWAKI_success_joint_gt.xlsx',
    # 'tracker_gt_output_results/vis_sub4_2_YAMAWAKI_success_joint_gt.xlsx',
    # 'tracker_gt_output_results/vis_sub4_3_YAMAWAKI_fail_joint_gt.xlsx',
    # 'tracker_gt_output_results/vis_sub4_1_TKATCHEV_success_joint_gt.xlsx',
    # 'tracker_gt_output_results/vis_sub4_2_TKATCHEV_success_joint_gt.xlsx',
    # 'tracker_gt_output_results/vis_sub4_3_TKATCHEV_success_joint_gt.xlsx',
]


sub1_data = []

# Read and reverse each file's data
for filepath in sub1_filepaths:
    df = pd.read_excel(filepath, skiprows=[0])
    df = df.dropna()
    df['t'] = pd.to_numeric(df['t'], errors='coerce')
    shoulder_angle = df.iloc[:, 2]
    hip_angle = df.iloc[:, 4]
    sub1_data.append(pd.concat([shoulder_angle, hip_angle], axis=1).iloc[::-1])
    

for  i,data in enumerate(sub1_data):
    print(len(data))
    sub1_data[i]['jointangle'] = moving_average_filter(data['jointangle'],5)
    sub1_data[i]['jointangle.1'] = moving_average_filter(data['jointangle.1'],5)
    

# Find the minimum data length
min_length = min(len(df) for df in sub1_data)

# Align to the minimum data length
sub1_aligned = [df.head(min_length) for df in sub1_data]

# Reverse the data back to the original order and reset index
sub1_aligned_reversed = [df.iloc[::-1].reset_index(drop=True) for df in sub1_aligned]

# Plot the graph using sub1_aligned_reversed
plt.figure()
hip_color = 'r'  # 指定 hip_angle 的顏色
shoulder_color = 'b'  # 指定 shoulder_angle 的顏色

plt.plot([], color=shoulder_color, label='Shoulder Angle')
plt.plot([], color=hip_color, label='Hip Angle')


for i in range(0, len(sub1_aligned_reversed)):
    t_values = sub1_aligned_reversed[i].index  # Get the reversed index as the 't' values
    
    time = [i/120 for i in range(len(t_values))]
    hip_color = 'r'  # 指定 hip_angle 的顏色
    shoulder_color = 'b'  # 指定 shoulder_angle 的顏色
    plt.plot(time, sub1_aligned_reversed[i].iloc[:, 0], color=shoulder_color)
    plt.plot(time, sub1_aligned_reversed[i].iloc[:, 1], color=hip_color)

plt.xlabel('Time')
plt.ylabel('Joint Angle')
plt.title('Sub3 Joint Angle')
plt.legend(loc='lower left')
plt.show()
