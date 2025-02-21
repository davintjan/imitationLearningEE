import argparse
import copy
import os
# import pytorch3d.transforms
import sys
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
    model.load_state_dict(torch.load("runs_FFBC_model_epoch_18.pth"))
    model.eval()

    # Load test dataset and actual labels (acts)
    test_dataset = RelativePoseDataset(relative_directory="../../expert_ML/predict_dataset")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    actual_labels = []  # Store actual labels
    for _, labels in test_loader:
        actual_labels.append(labels.numpy())
        # print('A')
    # print(actual_labels)
    """ 
    Actual labels is a list timestep (1227) of ndarray (1,70) -> 70 is just action over the next 10 timesteps flattened. So it is x,y,z,qx,qy,qz,qw over 10 timesteps
    Plot actual labels 
    """
    # Convert list to a single numpy array
    actual_labels = np.concatenate(actual_labels, axis=0)

    x = actual_labels[:, 0::7]
    y = actual_labels[:, 1::7] # N timesteps x 10
    z = actual_labels[:, 2::7]
    qx = actual_labels[:, 3::7]
    qy = actual_labels[:, 4::7]
    qz = actual_labels[:, 5::7]
    qw = actual_labels[:, 6::7]

    # Predict and compare with acts

    predictions = []
    count = 0
    for inputs, _ in test_loader:
        preds = predict(inputs, model)
        predictions.append(preds)
        count +=1
        print(count)


    """ PLOT TRAJECTORIES """
    predictions = np.concatenate(predictions, axis=0)
    # Extracting each variable based on the specified indices
    x_pred = predictions[:, 0::7] # N timesteps x 10
    y_pred = predictions[:, 1::7] # N timesteps x 10
    z_pred = predictions[:, 2::7]
    qx_pred = predictions[:, 3::7]
    qy_pred = predictions[:, 4::7]
    qz_pred = predictions[:, 5::7]
    qw_pred = predictions[:, 6::7]

    # # Create plots for each variable
    # variables = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    # actuals = [x, y, z, qx, qy, qz, qw]
    # preds = [x_pred, y_pred, z_pred, qx_pred, qy_pred, qz_pred, qw_pred]
    #
    # plt.figure(figsize=(20, 14))
    #
    # for i, (var, actual, pred) in enumerate(zip(variables, actuals, preds)):
    #     plt.subplot(4, 2, i+1)
    #     for t in range(actual.shape[0]):
    #         plt.plot(np.arange(t, t + 10), actual[t, :], label=f'Actual Timestep {t}', color='blue')
    #         plt.plot(np.arange(t, t + 10), pred[t, :], label=f'Pred Timestep {t}', color='red')
    #     plt.title(f'{var} vs {var}_pred')
    #     plt.xlabel('Timestep')
    #     plt.ylabel(var)
    #
    # plt.tight_layout()
    # plt.show()

    """ PLOT TRAJ FOR ANY 1 OF THE OUTPUT """
    # # Create a plot
    # plt.figure(figsize=(12, 6))
    #
    # # Plot each row with overlapping
    # for t in range(z_pred.shape[0]):
    #     plt.plot(np.arange(t, t + 10), qz_pred[t, :], label=f'Timestep {t}')
    #     plt.plot(np.arange(t, t + 10), qz[t, :], label=f'Timestep {t}')
    #
    # # Add title and labels
    # plt.title('Overlapping Segments Plot')
    # plt.xlabel('Timestep')
    # plt.ylabel('Value')
    #
    # # Show the plot
    # plt.show()

    """ PLOT POINTS INSTEAD OF TRAJ """
    x = actual_labels[:, 0]
    y = actual_labels[:, 1] # N timesteps x 1
    z = actual_labels[:, 2]
    qx = actual_labels[:, 3]
    qy = actual_labels[:, 4]
    qz = actual_labels[:, 5]
    qw = actual_labels[:, 6]

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