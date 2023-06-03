# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

def plot_data(sub_num, order, action, result, start, end, fig_num):
    hip_path = f"/home/m11002125/ViTPose/output_results/sub{sub_num}/sub{sub_num}_{order}_{action}_{result}_hip.csv"
    shoulder_path = f"/home/m11002125/ViTPose/output_results/sub{sub_num}/sub{sub_num}_{order}_{action}_{result}_shoulder.csv"

    hip_data = pd.read_csv(hip_path)
    shoulder_data = pd.read_csv(shoulder_path)

    hip_plot_data = hip_data.loc[:, "y0000"].to_numpy()
    shoulder_plot_data = shoulder_data.loc[:, "y0000"].to_numpy()

    plt.figure(num = fig_num)
    plt.subplot(121)
    plt.plot(hip_plot_data[start:end])
    plt.plot(shoulder_plot_data[start:end])
    plt.ylabel('joint angles')
    plt.xlabel('frames')
    plt.legend([f'order {order} hip angle',f'order {order} shoulder angle'])
    plt.subplot(122)
    plt.plot(shoulder_plot_data[start:end], hip_plot_data[start:end])
    plt.ylabel('hip angle')
    plt.xlabel('shoulder angle')

if __name__ == "__main__":
    sub_num = 3
    action = "TKATCHEV"
    result = "success"

    plot_data(sub_num, 12, action, result, 570, 845, 1)
    plot_data(sub_num, 15, action, result, 665, 940, 1)
    plot_data(sub_num, 18, action, result, 595, 870, 1)

    result = "fail"
    plot_data(sub_num, 3, action, result, 1315, 1590, 2)
    plot_data(sub_num, 6, action, result, 578, 853, 2)
    plot_data(sub_num, 9, action, result, 580, 855, 2)

    plt.show()
