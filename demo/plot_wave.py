# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd


'''
read csv output from my gui and plot.
'''

def plot_data(start,end):
    
    hip_path = f"/home/m11002125/ViTPose/vis_results/8_hip.csv"
    shoulder_path = f"/home/m11002125/ViTPose/vis_results/8_shoulder.csv"

    hip_data = pd.read_csv(hip_path)
    shoulder_data = pd.read_csv(shoulder_path)

    hip_plot_data = hip_data.loc[:, "y0000"].to_numpy()
    shoulder_plot_data = shoulder_data.loc[:, "y0000"].to_numpy()
    
    time = [i/60 for i in range(len(hip_plot_data[start:end]))]
    
    plt.figure()
    # plt.subplot(121)
    plt.plot(time,hip_plot_data[start:end])
    plt.plot(time,shoulder_plot_data[start:end])
    plt.ylabel('joint angles')
    plt.xlabel('time')
    
    plt.legend([f'hip angle',f'shoulder angle'],loc='lower left')
    
    # plt.subplot(122)
    # plt.plot(shoulder_plot_data[start:end], hip_plot_data[start:end])
    # plt.ylabel('hip angle')
    # plt.xlabel('shoulder angle')

if __name__ == "__main__":

    plot_data( 528, 645)


    plt.show()
