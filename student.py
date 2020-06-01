# TODO: use only one (RGB) channel
import numpy as np
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

dataset_path = "/home/igor/mlprojects/Csgo-NeuralNetwork/output/"
# train_split and test_split 0.1 > x > 0.9 and must add up to 1
train_split = 0.7
test_split = 0.3
num_epochs = 20
batch_size = 5

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s" % (torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')


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
        # dictionary that marks what the last frame of each folder is
        # ie. number of examples in specific folder
        self.folder_system = {}

        for folder in os.listdir(dataset_path):
            folder_len = 0
            for file in os.listdir(dataset_path + folder):
                folder_len += 1
            self.folder_system[folder_len] = '%s' % (folder)

        for folder_len in self.folder_system:
            self.length += folder_len
        self.length = int(self.length / 2)
        print(self.length)

    # returns name of folder that contains specific frame
    def find_folder(self, idx):
        for num_frames in self.folder_system:
            if num_frames >= idx:
                return str(self.folder_system[num_frames])
            else:
                idx = idx - num_frames

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
        with open(label_path) as file:
            # if file has important data
            if os.stat(label_path).st_size != 0:
                label = np.genfromtxt(file, delimiter=',')
                if np.shape(label) == (4,):
                    label = np.reshape(label, (1, 4))
                label = label[0:1, 0:4]
                label_shape = np.shape(label)
                label = torch.as_tensor(label, dtype=torch.long)
                label_shape = np.shape(label)
            # if file is blank (no data in the image), create -1 matrix
            else:
                label = torch.zeros([1, 4], dtype=torch.float)
                label[label == 0] = -1

            sample = {'image': img, 'label': label}

        # apply transforms
        # TODO: farofa aqui hein
        if self.transform:
            img = self.transform(sample['image'])
            img = img.reshape(172800)
            sample['image'] = img

        return sample

class CsgoClassificationDataset(data.Dataset):
    """
    Dataset extracts image label pairs from multiple folders and applies transforms to image.
    if labels are empty: creates 

    label format ==> [likelyhood there is a person]
                    so either [1] if there is a person
                    or [0] if there is none.
    """
    
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
        self.folder_system = {}

        for folder in os.listdir(dataset_path):
            folder_len  = 0
            for file in os.listdir(dataset_path + folder):
                folder_len += 1
            self.folder_system[folder_len] = '%s'%(folder)

        for folder_len in self.folder_system:
            self.length += folder_len
        self.length = int(self.length / 2)
        print(self.length)
        
    #returns name of folder that contains specific frame
    def find_folder(self, idx):
        for num_frames in self.folder_system:
            if num_frames >= idx:
                return str(self.folder_system[num_frames])
            else:
                idx = idx - num_frames

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
            #create [0] or [1] tensor
            label = torch.as_tensor((os.stat(label_path).st_size != 0), dtype=torch.long)
            sample = {'image':img, 'label':label}
            
        #apply transforms
        #TODO: farofa aqui hein
        if self.transform:
            img = self.transform(sample['image'])
            # img = img.reshape(172800)
            sample['image'] = img

        return sample

#defining NN layeres
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.last_linear = nn.Linear(in_features=32 * 80 * 45, out_features=2)

    def forward(self, x):
        # X comes in with shape(batch_size, 3, 320, 180)
        x = self.conv1(x)
        # After conv1 goes to shape (batch_size, 256, 320, 180)
        x = self.bn1(x)
        # After bn1 goes to the same shape (batch_size, 256, 320, 180)
        x = self.mp1(x)
        # After mp1 goes to the same shape (batch_size, 256, 160, 90)
        x = F.relu(x)
        # After relu it does not change shape (activations do not change shape)

        x = self.conv2(x)
        # After conv2 goes to shape (batch_size, 32, 160, 90)
        x = self.bn2(x)
        # After bn2 goes to shape (batch_size, 32, 160, 90)
        x = self.mp2(x)
        # After mp2 goes to shape (batch_size, 32, 80, 45)
        x = F.relu(x)
        # After relu it does not change shape (activations do not change shape)

        x = x.view(-1, 32 * 80 * 45)
        # After this view, it goes to shape (batch_size, 115200)
        x = self.last_linear(x)
        # After last_linear, it goes to shape (batch_size, 2)
        return x


# runs NN in training mode
def train_run(train_loader, criterion, optimizer, device):
    loss_record = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['label']
            # sends batch to gpu
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            print(loss.item())
            # print(labels)
            loss_record.append(loss.item())

            loss.backward(create_graph=False)

            optimizer.step()

            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

    print('Finished Training')
    plt.plot(loss_record)
    plt.show()

net = Net().to(device)

transform = transforms.Compose([
    transforms.Resize([320, 180]),
    # transforms.Resize([57600, 1]),
    transforms.ToTensor(),
])

dataset = CsgoClassificationDataset(dataset_path, transform)

dataset_len = len(dataset)

train_split = int(np.floor(dataset_len * train_split))
test_split = int(np.floor(dataset_len * test_split))
while train_split + test_split != dataset_len:
    train_split += 1
train_set, test_set = torch.utils.data.random_split( \
    dataset, [train_split, test_split])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters())
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

# for i in range(500):
#     image, label = dataset[i]['image'], dataset[i]['label']
#     print(label)

train_run(train_loader, criterion, optimizer, device)