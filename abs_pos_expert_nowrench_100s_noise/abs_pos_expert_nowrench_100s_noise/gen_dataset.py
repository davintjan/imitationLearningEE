import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torch.profiler import profile, record_function, ProfilerActivity
import os
import pickle
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from itertools import chain

""" TO DO INDEXING, JUST CALC THE INV WHEN IT IS CALLED """
""" WHEN INV, make a 4x4 and then just INV with linalg inv """
""" RN EVERYTIME WE GETITEM, WE RECALC EVERYTHING (ACTION WISE) """
class RelativePoseDataset(Dataset):
    def __init__(self, num_past_timestamps=100, num_future_timestamps= 100, relative_directory='../../expert_ML/train_1') -> None:
        """
        RETURN TWO THINGS:
        OBS: LIST (N_TRAJ) OF TRAJ (N_TIME X 7) [:-1] for every traj
        ACTION: LIST (N_TRAJ) OF TRAJ (N_TIME X 7) [:1] for every traj
        """
        super(RelativePoseDataset).__init__()
        self.num_past_timestamps = num_past_timestamps
        self.num_future_timestamps = num_future_timestamps
        self.base_directory = os.path.join(os.path.dirname(__file__), relative_directory)
        self.idx = ['target_poses', 'target_rel_poses', 'actual_poses', 'actual_rel_poses', 'goal_poses', 'wrench',
                    'joint_pos', 'joint_velo']
        self.keys_for_traj = ['bracket_in_anchor_channel_frame', 'target_wrench']
        self.trajectories = []
        self.filenames = []
        self.data = []
        self.len_idx = []
        self.cummulative_len_idx = []
        self.read_all_pickles(self.base_directory)

    def _read_from_pickle(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            self.data.append(data[0])

            actual_rel_pose = data[0][3]

            # Define mean and standard deviation for Gaussian offsets
            trans_mean = [0, 0, 0]  # Mean translation
            trans_std_dev = [0.001, 0.001, 0.001]  # Standard deviation for translation offsets

            orient_mean = [0, 0, 0]  # Mean orientation in degrees
            orient_std_dev = [0.5, 0.5, 0.5]  # Standard deviation for orientation offsets in degrees

            # Generate random offsets
            trans_offsets = np.random.normal(trans_mean, trans_std_dev)
            orient_offsets = np.random.normal(orient_mean, orient_std_dev)

            # Create offset matrix
            offset = np.hstack((trans_offsets, orient_offsets))
            offset_mat = self.offset_to_matrix(offset)

            noise_std_dev = 0.0005

            # For every item in the list, multiply with offset mat
                    # Apply the offset matrix to each pose in actual_rel_pose
            offset_applied_pose = []
            for pose_matrix in actual_rel_pose:
                offset_applied_matrix = offset_mat @ pose_matrix  # Matrix multiplication
                # Generate Gaussian noise
                noise = np.random.normal(0, noise_std_dev, offset_applied_matrix.shape)
            
                # Add noise to the offset-applied matrix
                noisy_pose_matrix = offset_applied_matrix + noise
                offset_applied_pose.append(noisy_pose_matrix)

            wrench_data = np.array(data[0][5])  # This is N timestep x 6
            trajectory = {
                'obs_wrench': wrench_data,  # List of N timesteps x 7 (x, y, z, qx, qy, qz, qw)
                'obs_pose': offset_applied_pose  # List of N timesteps x 6 (assuming wrench data has 6 components)
            }
            self.trajectories.append(trajectory) # F
            self.filenames.append(path)

    def read_all_pickles(self, directory):
        filenames = sorted(f for f in os.listdir(directory) if f.endswith('.pickle') or f.endswith('.pkl'))
        for filename in filenames:
            file_path = os.path.join(directory, filename)
            self._read_from_pickle(file_path)
        self._compute_cumulative_lengths()

    def offset_to_matrix(self, offset):
        """
        Input: a list of (6,) of translation(xyz), orientation(rpy)
        ORIENTATION IS IN DEGREE
        """
        # Extract translation and orientation from the offset list
        trans_offset = np.array(offset[0:3])
        orientation_offset = np.array(offset[3:])

        # Create a Rotation object from the RPY angles
        r = Rotation.from_euler('xyz', orientation_offset, degrees=True)

        # Get the rotation matrix
        R_matrix = r.as_matrix()

        # Construct the 4x4 transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R_matrix
        T[0:3, 3] = trans_offset
        
        return T

    def matrix_to_pose(self, matrix):
        translation = matrix[:3, 3]
        rotation = Rotation.from_matrix(matrix[:3, :3])
        quaternion = rotation.as_quat()
        return np.hstack((translation, quaternion))
    def _compute_cumulative_lengths(self):
        count = 0
        for traj in self.trajectories:
            length = len(traj['obs_wrench'])
            self.len_idx.append(length)
            count += length
            self.cummulative_len_idx.append(count)


    def __len__(self):
        return self.cummulative_len_idx[-1] if self.cummulative_len_idx else 0

    def __getitem__(self, idx):
        traj_idx = next(i for i, cumulative_len in enumerate(self.cummulative_len_idx) if idx < cumulative_len)
        local_idx = idx if traj_idx == 0 else idx - self.cummulative_len_idx[traj_idx - 1]
        data = self.trajectories[traj_idx]
        """ OBSERVATION """
        # If local_idx - (num_past_timestamps - 1) < 0, use np.tile to fill the first few missing elements
        start_idx = local_idx - (self.num_past_timestamps - 1)
        if start_idx < 0:
            num_missing = -start_idx
            obs_mat_start = np.tile(data['obs_pose'][0], (num_missing, 1, 1))
            obs_mat_rest = np.array(data['obs_pose'][:local_idx + 1])
            obs_mat = np.concatenate((obs_mat_start, obs_mat_rest), axis=0)

        else:
            obs_mat = np.array(data['obs_pose'][start_idx:local_idx + 1]) # This is 20 x 4 x 4 where 4x4 is the transformation matrix

        # Change to pose use matrix to pose
        obs_pose = [self.matrix_to_pose(matrix) for matrix in obs_mat]
        obs = obs_pose

        """ ACTION """
        end_idx = local_idx + self.num_future_timestamps + 1
        if end_idx > len(data['obs_pose']):
            # If end_idx is beyond the length of data, we need to handle it carefully
            if local_idx + 1 < len(data['obs_pose']):
                act_mat_before_end = np.array(data['obs_pose'][local_idx + 1:])
                act_mat_end = np.tile(data['obs_pose'][-1], (end_idx - len(data['obs_pose']), 1, 1))
                act_mat = np.concatenate((act_mat_before_end, act_mat_end), axis=0)
            else:
                # Handle edge case where local_idx is the last index
                act_mat = np.tile(data['obs_pose'][-1], (self.num_future_timestamps, 1, 1))
        else:
            act_mat = np.array(data['obs_pose'][local_idx + 1:end_idx])

        act_pose = [self.matrix_to_pose(matrix) for matrix in act_mat]
        act = act_pose

        return np.array(obs).flatten(), np.array(act).flatten()

if __name__ == "__main__":

    # Create an instance of RelativePoseDataset
    dataset = RelativePoseDataset()

    g = dataset.__getitem__(0)
    print(g[0].shape, g[1].shape)




