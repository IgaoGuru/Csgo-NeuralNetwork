from torch.utils.data.dataloader import DataLoader
from os import replace
from torchvision import transforms
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from os import walk
import torchvision
import cv2
import numpy as np
import torch.nn as nn
import pickle
# Our own libraries
from stat_interpreter_reg import interpreter
import fastercnn
import datasetcsgo
import autoencoder
from util import loss_dict_template

# Our own libraries
import datasetcsgo

class AE(nn.Module):
    def __init__(self, input_shape, stride, kernel_size, padding, num_out_channels, device):
        super().__init__()
        self.stride, self.kernel_size, self.padding, self.num_out_channels = stride, kernel_size, padding, num_out_channels

        # self.nn_layers = nn.ModuleList()
        self.conv1 = self.encoder_hidden_layer = nn.Conv2d(
            3, num_out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, bias=False),

    def forward(self, features):
        self.conv1(features)

        x = self.conv1(x)
        x = nn.BatchNorm2d(num_out_channels),
        x = nn.ReLU(inplace=True)
        print(np.Size(features))

        return reconstructed

dataset_path = "/home/igor/mlprojects/Csgo-NeuralNetworkold/data/datasets/"  #remember to put "/" at the end
model_save_path = '/home/igor/mlprojects/ae_modelsave/'
SEED = 42

num_epochs = 10
scale_factor = 1
batch_size = 1

#MODEL PARAMS
stride, kernel_size, padding, num_out_channels = 2, 3, 2, 3

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')

transform = transforms.Compose([
    transforms.Resize([int(1080*scale_factor), int(1920*scale_factor)]),
    transforms.ToTensor(), # will put the image range between 0 and 1
])

def my_collate_2(batch):
    imgs = [item[0] for item in batch]
    bboxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return [imgs, bboxes, labels]

dataset = datasetcsgo.CsgoDataset(dataset_path, transform=transform, scale_factor=scale_factor)
train_set, val_set, _ = dataset.split(train=0.7, val=0.15, seed=SEED)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)

model = AE(input_shape = int(1080*1920), stride=2, kernel_size=kernel_size, padding=padding, num_out_channels=num_out_channels, device=device)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
log_interval = len(train_loader) // 1
log_interval_val = len(val_loader) // 1

arguments = {
    'input_shape' : 1080*1920,
    'device' : device
}


def train_cycle_ae():
    ae_loss_dict_total = {'loss_ae'}
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(dataset):
            imgs, _, _ = data 
            print(np.shape(imgs))

            # optimizer.zero_grad()

            # outputs = model()

train_cycle_ae()