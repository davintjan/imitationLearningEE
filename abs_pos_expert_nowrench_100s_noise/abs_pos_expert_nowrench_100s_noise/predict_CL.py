import argparse
import copy
import os
# import pytorch3d.transforms
import sys

import numpy as np
import torch
from datetime import datetime
from enum import Enum
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from gen_dataset import *
from Model import *
import random
from torch.cuda import is_available as cuda_available
from matplotlib import pyplot as plt
from tqdm import tqdm

def predict(input_data, model):
    model.eval()
    inputs = torch.tensor(input_data).float()
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.numpy()

def main():
    model = FFBC()
    model.load_state_dict(torch.load("runs_FFBC_model_epoch_38.pth"))
    model.eval()

    # Load test dataset and actual labels (acts)
    test_dataset = RelativePoseDataset(relative_directory="../../expert_ML/predict_dataset")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    actual_labels = []  # Store actual labels
    for _, labels in test_loader:
        actual_labels.append(labels.numpy())
        # print(actual_labels)
    """ 
    Actual labels is a list timestep (1227) of ndarray (1,70) -> 70 is just action over the next 10 timesteps flattened. So it is x,y,z,qx,qy,qz,qw over 10 timesteps
    Plot actual labels 
    """
    # Convert list to a single numpy array
    actual_labels = np.concatenate(actual_labels, axis=0)

    """ PLOT POINTS INSTEAD OF TRAJ """
    x = actual_labels[:, 0]
    y = actual_labels[:, 1] # N timesteps x 1
    z = actual_labels[:, 2]
    qx = actual_labels[:, 3]
    qy = actual_labels[:, 4]
    qz = actual_labels[:, 5]
    qw = actual_labels[:, 6]

    predictions = []
    init_obs,_ = test_dataset.__getitem__(0) # This is (140,)
    count = 0

    for i in range(2000):
        preds = predict(init_obs, model) # This is 70,
        predictions.append(preds)
        immediate_next_step = preds[:7] # This is 7,
        # Remove first 7 element from init obs and append immediate next step to the last 7 element
        init_obs = np.append(init_obs, immediate_next_step)
        init_obs = init_obs[7:]
        count +=1
        print(count)
        
    predictions = np.array(predictions)
    x_pred = predictions[:, 0] # N timesteps x 1
    y_pred = predictions[:, 1] # N timesteps x 1
    z_pred = predictions[:, 2]
    qx_pred = predictions[:, 3]
    qy_pred = predictions[:, 4]
    qz_pred = predictions[:, 5]
    qw_pred = predictions[:, 6]

    # Create plots for each variable
    variables = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    actuals = [x, y, z, qx, qy, qz, qw]
    preds = [x_pred, y_pred, z_pred, qx_pred, qy_pred, qz_pred, qw_pred]

    plt.figure(figsize=(20, 14))

    for i, (var, actual, pred) in enumerate(zip(variables, actuals, preds)):
        plt.subplot(4, 2, i + 1)
        plt.plot(range(len(actual)), actual, label=f'Actual {var}', color='blue', linewidth=1)
        plt.plot(range(len(pred)), pred, label=f'Predicted {var}', color='red', linewidth=1)
        plt.title(f'{var} vs {var}_pred')
        plt.xlabel('Timestep')
        plt.ylabel(var)
        if i == 0:  # Only add legend to the first plot to avoid clutter
            plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    print("a")