import warnings
warnings.simplefilter(action='ignore')
import os
from matplotlib import pylab as plt
import h5py
from torch.autograd import Variable
import torch.nn.init
from torch.utils.data import Dataset, DataLoader
from MMDF_ESI_functions import input_reshape, input_padding, RandomDatasetMMDF, MMDFESI
from argparse import ArgumentParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('==================================== device:', device, '====================================')

parser = ArgumentParser(description='MMDFESI')
parser.add_argument('--print_model_flag', type=int, default=0, help='print structure of model')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
parser.add_argument('--save_freq', type=float, default=50, help='save frequency of model')
parser.add_argument('--batch_size', type=int, default=500, help='batch size of training data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--change_loss', type=float, default=10, help='when learning rate changed, save best model according to this loss value')

args = parser.parse_args()
start_epoch = args.start_epoch
end_epoch = args.end_epoch
batch_size = args.batch_size
save_freq = args.save_freq
change_loss = args.change_loss
learning_rate = args.learning_rate
print_model_flag = args.print_model_flag

# Set FC module parameters
fc1_num = 1920
fc2_num = 500
output_num = 1984

# Set path
maptable_eeg = './data/eeg_maptable.mat'
maptable_meg = './data/meg_maptable.mat'
data_fname = './data/MMDFESI_sample_data.mat'

model_dir = './model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# Load dataset
dataset = h5py.File(data_fname)
train_output = dataset['source_train'][()]
train_input_eeg = dataset['EEG_train'][()]
train_input_meg = dataset['MEG_train'][()]

# Data reshape & padding
train_input_matrix_eeg = input_reshape(train_input_eeg, maptable_eeg)
train_input_matrix_meg = input_reshape(train_input_meg, maptable_meg)
train_input_matrix_eeg, train_input_matrix_meg = input_padding(train_input_matrix_eeg, train_input_matrix_meg)

print('check input shape:')
print(train_input_matrix_eeg.shape)
print(train_input_matrix_meg.shape)

# Data loader
RandomDataset_train = RandomDatasetMMDF(train_input_matrix_eeg,
                                        train_input_matrix_meg,
                                        train_output)
Rand_loader = DataLoader(dataset=RandomDataset_train,
                               batch_size=batch_size,
                               num_workers=0,
                               shuffle=True)

# Build model
model = MMDFESI(fc1_num=fc1_num, fc2_num=fc2_num, output_num=output_num)
model = model.to(device)
if print_model_flag == 1:
    print(model)
model = model.to(device)

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('%s/net_params_best.pkl' % (model_dir)))

if start_epoch > 0:
    loss_threshold = change_loss
else:
    loss_threshold = 9999


criterion_MSE = torch.nn.MSELoss()
optimizer = torch.optim.NAdam(params=model.parameters(), lr=learning_rate)


# training loop
print('===================== Training Start: =========================')
loss_mse_train_save = []

for epoch in range(start_epoch + 1, end_epoch + 1):
    loss_mse_train = 0

    for data in Rand_loader:
        batch_X_eeg, batch_X_meg, batch_Y, i = data
        batch_X_eeg = batch_X_eeg.to(device)
        batch_X_meg = batch_X_meg.to(device)
        batch_Y = batch_Y.to(device)

        X_eeg = Variable(batch_X_eeg)
        X_meg = Variable(batch_X_meg)
        Y = Variable(batch_Y)

        # initialization of the gradients
        optimizer.zero_grad()

        # forward propagation
        Y_pred = model(X_eeg, X_meg)
        loss_mse = criterion_MSE(Y_pred, Y)

        # compute the loss function
        loss = loss_mse

        # backward propagation
        loss.backward()
        optimizer.step()

        loss_mse_train += loss_mse # tensor

    if loss_mse_train < loss_threshold:
        torch.save(model.state_dict(), "%s/net_params_best.pkl" % (model_dir))
        loss_threshold = loss_mse_train
        print("best model in epoch:", epoch)

    loss_mse_train_save.append(loss_mse_train.item())

    output_loss_iter = "[%02d/%02d] Training Loss: %.4f\n" % (epoch, end_epoch, loss_mse_train.item())

    if epoch % save_freq == 0:
        torch.save(model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch))  # save only the parameters

    print(output_loss_iter)


plt.figure(1)
plt.plot(loss_mse_train_save, label='Train')
plt.title('loss during iteration', fontsize='12')
plt.ylabel('loss value', fontsize='10')
plt.xlabel('iteration', fontsize='10')
plt.legend()
plt.savefig(model_dir + '/Train_loss_mse.png')


print('===================== Training Finished! =========================')

