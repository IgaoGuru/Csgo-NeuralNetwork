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
from util import loss_dict_template

# Our own libraries
import datasetcsgo

class AE(nn.Module):
    def __init__(self, stride, kernel_size, padding, num_out_channels, correction):
        super(AE, self).__init__()
        # stride, kernel_size, padding, num_out_channels = self.stride, self.kernel_size, self.padding, self.num_out_channels
        self.stride, self.kernel_size, self.padding, self.num_out_channels, self.correction = stride, kernel_size, padding, num_out_channels, correction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size, stride=stride, padding=padding),  # b, 16, 10, 10
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5

            nn.Conv2d(16, 32, kernel_size, stride=stride, padding=padding),  # b, 8, 3, 3
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2

            nn.Conv2d(32, 32, kernel_size, stride=stride, padding=padding),  # b, 8, 3, 3
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        if correction == True:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size, stride=stride, padding=padding),  # b, 16, 5, 5
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, kernel_size, stride=stride, padding=1),  # b, 8, 15, 15
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 3, kernel_size, stride=stride, padding=padding),  # b, 1, 28, 28
                nn.ZeroPad2d((1, 0, 1, 0))
            # nn.Tanh()
            )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size, stride=stride, padding=padding),  # b, 16, 5, 5
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, kernel_size, stride=stride, padding=1),  # b, 8, 15, 15
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 3, kernel_size, stride=stride, padding=padding),  # b, 1, 28, 28
            )

    def forward(self, x):
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        return x

dataset_path = "/home/igor/mlprojects/Csgo-NeuralNetworkold/data/datasets/"  #remember to put "/" at the end
model_save_path = '/home/igor/mlprojects/modelsave-autoencoder/'
SEED = 42

model_number = 1
num_epochs = 200
scale_factor = 1
batch_size = 1

#MODEL PARAMS
stride, kernel_size, padding, num_out_channels = 2, 3, 2, 3 #optimal: 2323
# 2513 also works

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

arguments = {
    'input_shape' : 1080*1920,
    'device' : device
}


def train_cycle_ae():
    dataset = datasetcsgo.CsgoDataset(dataset_path, transform=transform, scale_factor=scale_factor)
    print(len(dataset))
    train_set, val_set, _ = dataset.split(train=0.7, val=0.15, seed=SEED)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    model = AE(stride=stride, kernel_size=kernel_size, padding=padding, num_out_channels=num_out_channels, correction=correction)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    log_interval = len(train_loader) // 1
    print(f'log_interval is: {log_interval}')
    log_interval_val = len(val_loader) // 1
    print(f'log_interval_val is {log_interval_val}')

    model_save_path_new = f"{model_save_path}aemodel#{model_number}"  
    ae_loss_dict = {'loss_ae':[], 'loss_ae_val':[]}
    total_i = len(train_loader)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_loss_val = 0.0
        for i, data in enumerate(train_loader):
            imgs, _, _ = data 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            imgs = imgs.to(device)

            # ===================forward=====================
            output = model(imgs)
            
            img = output[0].detach().cpu().numpy().copy().transpose(1, 2, 0)
            cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            loss = criterion(output, imgs)
            running_loss += loss.item()
            
            if (i + 1) % log_interval == 0:
                print('%s ::Training:: [%d, %5d] loss: %.5f' %
                    ('yeah yeah', epoch + 1, i + 1, running_loss / log_interval))
                ae_loss_dict['loss_ae'].append(running_loss)
                running_loss = 0.0
        
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #pytorch lindinho

        for i, data in enumerate(val_loader):

            imgs = imgs.to(device)

            # ===================forward=====================
            output = model(imgs)
            
            img = output[0].detach().cpu().numpy().copy().transpose(1, 2, 0)
            cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            loss = criterion(output, imgs)
            running_loss_val += loss.item()
            
            if (i + 1) % log_interval == 0:
                print('%s ::Validation:: [%d, %5d] loss: %.5f' %
                    ('yeah yeah', epoch + 1, i + 1, running_loss_val / log_interval_val))
                ae_loss_dict['loss_ae_val'].append(running_loss_val)
                running_loss_val = 0.0
                if epoch in checkpoints: 
                    print(f"Saving net at: {model_save_path_new}")
                    torch.save(model.state_dict(), model_save_path_new + 'e' + f'{epoch}' + '.th')
                    with open(f'{model_save_path_new}-train', 'wb') as filezin:
                        pickle.dump(ae_loss_dict, filezin)

checkpoints = [0, 1, 2, 15, 29, 49, 99, 149, 199]

model_number = 1
stride, kernel_size, padding, num_out_channels = 2, 2, 2, 3 #optimal: 2323
correction = False
train_cycle_ae()

model_number = 2
stride, kernel_size, padding, num_out_channels = 2, 3, 2, 3 #optimal: 2323
correction = True
train_cycle_ae()

model_number = 3
stride, kernel_size, padding, num_out_channels = 2, 5, 1, 3 #optimal: 2323
correction = True
train_cycle_ae()