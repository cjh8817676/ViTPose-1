# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 定義數據
data = {
    "Sub1": {
        "YAMAWAKI": {
            1: [129.859, 126.054],
            2: [126.768, 118.952],
            3: [129.297, 126.365],
            4: [136.287, 128.119],
            5: [131.461, 130.251],
            6: [138.321, 116.339]
        }
    },
    "Sub2": {
        "YAMAWAKI": {
            1: [132.389, 133.345],
            2: [132.389, 131.013],
            3: [137.168, 140.766],
            4: [129.478, 134.172],
            5: [141.711, 144.549],
            6: [137.156, 139.448]
        }
    },
    "Sub3": {
        "TKATCHEV": {
            4: [147.146, 139.244],
            5: [144.476, 139.040],
            6: [140.259, 134.129],
            7: [147.046, 135.915],
            8: [148.217, 136.311],
            9: [148.087, 138.455],
            10: [147.436, 137.569],
            12: [146.265, 138.088]
        }
    },
    "Sub4": {
        "TKATCHEV": {
            1: [153.162, 145.781],
            2: [150.820, 143.412],
            3: [155.765, 150.398]
        },
        "YAMAWAKI": {
            1: [190.899, 184.960],
            2: [197.721, 186.728]
        }
    }
}

maes = []
rmses = []

# 計算每個SUBJECT的MAE和RMSE
for subject, trials in data.items():
    subject_mae = 0
    subject_rmse = 0

    for trial, values in trials.items():
        gt = np.array([value[0] for value in values.values()])
        pred = np.array([value[1] for value in values.values()])

        mae = mean_absolute_error(gt, pred)
        rmse = np.sqrt(mean_squared_error(gt, pred))

        subject_mae += mae
        subject_rmse += rmse

    subject_mae /= len(trials)
    subject_rmse /= len(trials)

    maes.append(subject_mae)
    rmses.append(subject_rmse)

# 計算總MAE和RMSE
total_mae = np.mean(maes)
total_rmse = np.mean(rmses)

print("每個SUBJECT的MAE:", maes)
print("每個SUBJECT的RMSE:", rmses)
print("總MAE:", total_mae)
print("總RMSE:", total_rmse)

