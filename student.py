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
import tqdm


dataset_path = "/home/igor/mlprojects/Csgo-NeuralNetwork/output/"
#train_split and test_split 0.1 > x > 0.9 and must add up to 1
train_split = 0.7
test_split = 0.3
num_epochs = 2
batch_size = 5
# NOTE: possibly remove in next verssion??
# video_folder_name = "CSGOraw1"
# label_index = 12
# label_path = dataset_path + video_folder_name +'/'+ ("%sframe#%s" % (video_folder_name, label_index)) + '.txt'
# bboxes = pd.read_csv(label_path)

class CsgoPersonDataset(data.Dataset):

    """preety description."""

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
        #dictionary that marks what the last frame of each folder is
        #ie. number of examples in specific folder
        self.folder_system = {10167:'CSGOraw1'}

        for folder_index in self.folder_system:
            self.length += folder_index
        
    #returns name of folder that contains specific frame
    def find_folder(self, idx):
        for num_frames in self.folder_system:
            if num_frames >= idx:
                return str(self.folder_system[num_frames])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #sets path and gets txt/jpg files
        img_path = self.find_folder(idx)
        img_name = "%sframe#%s" % (img_path, idx)
        img_path = os.path.join(self.root_dir,
                                img_path, img_name)
        img_path_ext = img_path + '.jpg'
        img = Image.open((img_path_ext))
        # img = np.array(img)
        label_path = str(img_path) + '.txt'
        label = 0
        #loads label from disk, converts csv to tensor
        with open(label_path) as file:
            #if file has important data
            if os.stat(label_path).st_size != 0:
                label = np.genfromtxt(file, delimiter=',')
                if np.shape(label) == (5,):
                    label = np.reshape(label, (1, 5))
                label_shape = np.shape(label)
                label = label[0:1, 0:5]
                label = torch.as_tensor(label, dtype=torch.float)
                label_shape = np.shape(label)
            #if file is blank (no data in the image), create -1 matrix
            else:
                label = torch.zeros([1, 5], dtype=torch.float)
                label[label==0] = -1
            
            sample = {'image':img, 'label':label}
            
        #apply transforms
        #TODO: farofa aqui hein
        if self.transform:
            img = self.transform(sample['image'])
            img = img.reshape(172800)
            sample['image'] = img

        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3*320*180, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


transform = transforms.Compose([
    transforms.Resize([320, 180]),
    transforms.Resize([57600, 1]),
    transforms.ToTensor()
])

dataset = CsgoPersonDataset(dataset_path, transform)

dataset_len = len(dataset)

train_split = int(np.floor(dataset_len * train_split))
test_split = int(np.floor(dataset_len * test_split))
while train_split + test_split != dataset_len:
    train_split += 1
train_set, test_set = torch.utils.data.random_split(\
    dataset, [train_split, test_split])
    
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=4)

classes = ('background', 'enemy')

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(tqdm.tqdm(train_loader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'], data['label']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

print('Finished Training')