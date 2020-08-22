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
    def __init__(self, stride, kernel_size, padding, num_out_channels):
        super(AE, self).__init__()
        # stride, kernel_size, padding, num_out_channels = self.stride, self.kernel_size, self.padding, self.num_out_channels
        self.stride, self.kernel_size, self.padding, self.num_out_channels = stride, kernel_size, padding, num_out_channels 
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=stride, padding=padding),  # b, 16, 10, 10
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5

            nn.Conv2d(16, 32, 3, stride=stride, padding=padding),  # b, 8, 3, 3
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2

            nn.Conv2d(32, 32, 3, stride=stride, padding=padding),  # b, 8, 3, 3
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=stride, padding=padding),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=stride, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=stride, padding=padding),  # b, 1, 28, 28
            nn.ZeroPad2d((1, 0, 1, 0))
            # nn.Tanh()
        )

    def forward(self, x):
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        return x

dataset_path = "/home/igor/mlprojects/Csgo-NeuralNetworkold/data/datasets/"  #remember to put "/" at the end
model_save_path = '/home/igor/mlprojects/ae_modelsave/'
SEED = 42

num_epochs = 10
scale_factor = 1
batch_size = 1

#MODEL PARAMS
stride, kernel_size, padding, num_out_channels = 2, 3, 0, 3

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
train_set, val_set, _ = dataset.split(train=0.1, val=0.15, seed=SEED)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

model = AE(stride=2, kernel_size=3, padding=2, num_out_channels=3)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

log_interval = len(train_loader) // 1
print(f'log_interval is: {log_interval}')
log_interval_val = len(val_loader) // 1

arguments = {
    'input_shape' : 1080*1920,
    'device' : device
}


def train_cycle_ae():
    ae_loss_dict_total = {'loss_ae'}
    total_i = len(train_loader)
    running_loss = 0.0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader):
            imgs, _, _ = data 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            imgs = imgs.to(device)

            # ===================forward=====================
            output = model(imgs)
            
            img = output[0].detach().cpu().numpy().copy().transpose(1, 2, 0)
            cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            loss = criterion(output, imgs)
            running_loss += loss.item()
            
            if (i + 1) % log_interval == 0:
                print('%s ::Training:: [%d, %5d] loss: %.5f' %
                    ('yeah yeah', epoch + 1, i + 1, running_loss / log_interval))
                running_loss = 0.0
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #pytorch lindinho

train_cycle_ae()