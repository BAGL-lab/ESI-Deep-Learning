import numpy as np
import h5py
import torch
import torch.nn.init
from torch.utils.data import Dataset


def input_reshape(data, fname):
    """
    change vector of EEG/MEG to matrix.
    """
    matfile = h5py.File(fname)
    maptable = matfile['maptable'][()].T
    x = max(maptable[:, 0]) + 1
    y = max(maptable[:, 1]) + 1
    data_num = data.shape[0]
    temp_matrix = np.zeros((data_num, int(x), int(y)))
    for index in range(maptable.shape[0]):
        i = maptable[index, 0]
        j = maptable[index, 1]
        value = maptable[index, 2]
        temp_matrix[:, int(i), int(j)] = data[:, int(value)]
    return temp_matrix


def input_padding(matrix_1, matrix_2):
    """
    change matrix of EEG/MEG to same shape.
    """
    nsample, row_1, col_1 = matrix_1.shape[0], matrix_1.shape[1], matrix_1.shape[2]
    row_2, col_2 = matrix_2.shape[1], matrix_2.shape[2]
    row_max = max(row_1, row_2)
    col_max = max(col_1, col_2)
    matrix_1_new = np.zeros(shape=(nsample, row_max, col_max))
    matrix_2_new = np.zeros(shape=(nsample, row_max, col_max))
    row_dif_1 = row_max-row_1
    col_dif_1 = col_max-col_1
    row_dif_2 = row_max-row_2
    col_dif_2 = col_max-col_2

    idx_row_1 = int(row_dif_1/2)
    idx_row_2 = int(row_dif_2/2)
    idx_col_1 = int(col_dif_1/2)
    idx_col_2 = int(col_dif_2/2)

    matrix_1_new[:, idx_row_1:idx_row_1+row_1, idx_col_1:idx_col_1+col_1] = matrix_1[:, :, :]
    matrix_2_new[:, idx_row_2:idx_row_2+row_2, idx_col_2:idx_col_2+col_2] = matrix_2[:, :, :]

    return matrix_1_new, matrix_2_new


class RandomDatasetMMDF(Dataset):
    def __init__(self, x_data_eeg, x_data_meg, y_data=None):
        self.x_data_eeg = x_data_eeg.reshape((x_data_eeg.shape[0], 1, x_data_eeg.shape[1], x_data_eeg.shape[2]))
        self.x_data_meg = x_data_meg.reshape((x_data_meg.shape[0], 1, x_data_meg.shape[1], x_data_meg.shape[2]))
        self.len = x_data_eeg.shape[0]

        if y_data is not None:
            self.y_data = y_data
        else:
            # create a dummy y_data placeholder with the same number of samples as x_data_eeg
            self.y_data = torch.zeros((x_data_eeg.shape[0], 1))

    def __getitem__(self, index):
        x_batch_eeg = torch.Tensor(self.x_data_eeg[index, :, :, :]).float()
        x_batch_meg = torch.Tensor(self.x_data_meg[index, :, :, :]).float()
        y_batch = torch.Tensor(self.y_data[index, :]).float()
        return x_batch_eeg, x_batch_meg, y_batch, index

    def __len__(self):
        return self.len


class MMDFESI(torch.nn.Module):
    def __init__(self, fc1_num, fc2_num, output_num):
        super(MMDFESI, self).__init__()
        self.cnn_eeg = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, dilation=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            )
        self.cnn_meg = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, dilation=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            )
        self.SE = torch.nn.Sequential(
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.Sigmoid(),
            )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(fc1_num, fc2_num, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(fc2_num, output_num, bias=True)
            )

    def forward(self, x_eeg, x_meg):
        feature_eeg = self.cnn_eeg(x_eeg)
        feature_meg = self.cnn_meg(x_meg)
        feature_eeg = feature_eeg.view(feature_eeg.size(0), feature_eeg.size(1), -1)
        feature_meg = feature_meg.view(feature_meg.size(0), feature_meg.size(1), -1)
        feature_concat = torch.cat([feature_eeg, feature_meg], dim=1)
        squeeze = torch.mean(feature_concat, dim=2)
        excitation = self.SE(squeeze)
        excitation = excitation.unsqueeze(2)
        scale_concat = torch.mul(feature_concat, excitation)
        feature_flat = scale_concat.view(scale_concat.size(0), -1)
        output = self.classifier(feature_flat)
        return output
