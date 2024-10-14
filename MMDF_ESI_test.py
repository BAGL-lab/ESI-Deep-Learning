import warnings
warnings.simplefilter(action='ignore')
import scipy.io as sio
import os
import h5py
from torch.autograd import Variable
import torch.nn.init
from torch.utils.data import Dataset, DataLoader
from MMDF_ESI_functions import input_reshape, input_padding, RandomDatasetMMDF, MMDFESI

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('==================================== device:', device, '====================================')

# Set FC module parameters
fc1_num = 1920
fc2_num = 500
output_num = 1984

# Set path
maptable_eeg = './data/eeg_maptable.mat'
maptable_meg = './data/meg_maptable.mat'
data_fname = './data/MMDFESI_sample_data.mat'

model_dir = './model/'
result_dir = './result/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

result_fname = result_dir + 'test_result.mat'

# Load dataset
dataset = h5py.File(data_fname)
test_input_eeg = dataset['EEG_test'][()]
test_input_meg = dataset['MEG_test'][()]

# Data reshape & padding
test_input_matrix_eeg = input_reshape(test_input_eeg, maptable_eeg)
test_input_matrix_meg = input_reshape(test_input_meg, maptable_meg)
test_input_matrix_eeg, test_input_matrix_meg = input_padding(test_input_matrix_eeg, test_input_matrix_meg)

print('check input shape:')
print(test_input_matrix_eeg.shape)
print(test_input_matrix_meg.shape)

# Data loader
RandomDataset_test = RandomDatasetMMDF(test_input_matrix_eeg,
                                        test_input_matrix_meg,
                                        y_data=None)
Rand_loader = DataLoader(dataset=RandomDataset_test,
                               batch_size=test_input_matrix_eeg.shape[0],
                               num_workers=0,
                               shuffle=True)


# Build model
model = MMDFESI(fc1_num=fc1_num, fc2_num=fc2_num, output_num=output_num)
model = model.to(device)
model.load_state_dict(torch.load('%s/net_params_best.pkl' % (model_dir)))

# Test
model.eval()
with torch.no_grad():
    for data in Rand_loader:
        batch_X_eeg, batch_X_meg, _, _ = data
        batch_X_eeg = batch_X_eeg.to(device)
        batch_X_meg = batch_X_meg.to(device)

        X_eeg = Variable(batch_X_eeg)
        X_meg = Variable(batch_X_meg)

        Y_pred = model(X_eeg, X_meg)
        Y_pred = Y_pred.cpu().detach().numpy().T

sio.savemat(result_fname, {'s_pred': Y_pred})
print("Test results saved in: {}".format(result_fname))

