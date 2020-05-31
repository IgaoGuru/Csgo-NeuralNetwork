#TODO: use only one (RGB) channel
import numpy as np
import pandas as pd
import os
from torch.utils import data
from torch.utils.data.dataloader import DataLoader as DataLoader
import torch
from torchvision import transforms
from natsort import natsorted, ns
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


dataset_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetwork\\output\\"
#train_split and test_split 0.1 > x > 0.9 and must add up to 1
train_split = 0.7
test_split = 0.3
num_epochs = 10
batch_size = 100

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')


class CsgoPersonNoPersonDataset(data.Dataset):
    """pretty description."""

    length = -1

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.6)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.length = 0
        # dictionary that marks what the last frame of each folder is
        # ie. number of examples in specific folder
        self.folder_system = {2426: 'CSGOraw2'}

        for folder_index in self.folder_system:
            self.length += folder_index

    # returns name of folder that contains specific frame
    def find_folder(self, idx):
        for num_frames in self.folder_system:
            if num_frames >= idx:
                return str(self.folder_system[num_frames])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sets path and gets txt/jpg files
        img_path = self.find_folder(idx)
        img_name = "%sframe#%s" % (img_path, idx)
        img_path = os.path.join(self.root_dir,
                                img_path, img_name)
        img_path_ext = img_path + '.jpg'
        img = Image.open((img_path_ext))
        # img = np.array(img)
        label_path = str(img_path) + '.txt'
        label = 0
        # loads label from disk, converts csv to tensor

        label = torch.as_tensor(os.stat(label_path).st_size != 0, dtype=torch.float).reshape((1,))
        sample = {'image': img, 'label': label}

        # apply transforms
        # TODO: farofa aqui hein
        if self.transform:
            img = self.transform(sample['image'])
            # img = img.reshape(172800)
            sample['image'] = img

        return sample

#defining NN layeres
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 61 * 33, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)
        self.fc4 = nn.Linear(30, 15)
        self.fc5 = nn.Linear(15, 7)
        self.fc6 = nn.Linear(7, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 33)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        #x = F.relu(self.fc6(x))
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)

#runs NN in training mode
def train_run(train_loader, criterion, optimizer, device):
    losses = []
    print(len(train_loader.dataset))
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['label']
            #if labels[0].item() == -1:
            #    continue
            #sends batch to gpu
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            #print(f"{epoch}, {i}")
            outputs = net(inputs)
            #print(f"Labels: {labels.shape}, {labels.dtype}")
            #print(f"Outputs: {outputs.shape}, {outputs.dtype}")

            loss = criterion(outputs, labels)
            losses.append(loss.item())
            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                print(f"Labels: {torch.transpose(labels, 0, 1)}")
                print(f"Outputs: {torch.transpose(outputs, 0, 1)}")
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
                print("-------------------------------------")

            loss.backward()
            optimizer.step()

    print('Finished Training')
    return losses

net = Net().to(device)
net.apply(weights_init)

transform = transforms.Compose([
    transforms.Resize([256, 144]),
    # transforms.Resize([57600, 1]),
    transforms.ToTensor(),
])

dataset = CsgoPersonNoPersonDataset(dataset_path, transform)

dataset_len = len(dataset)

train_split = int(np.floor(dataset_len * train_split))
test_split = int(np.floor(dataset_len * test_split))
while train_split + test_split != dataset_len:
    train_split += 1
train_set, test_set = torch.utils.data.random_split(\
    dataset, [train_split, test_split])
    
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True)

def my_binary_loss(output, target):
    return (output and target).mean

criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(net.parameters())



# for i in range(500):
#     image, label = dataset[i]['image'], dataset[i]['label']
#     print(label)

losses = train_run(train_loader, criterion, optimizer, device)
print("------------------------------------------------------------")
print("Losses")
for loss in losses:
    print(loss)
print("------------------------------------------------------------")