# TODO: use only one (RGB) channel
import numpy as np
from math import ceil
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
from time import time
import pickle
torch.manual_seed(42)

#CONTROL PANEL -------------
dataset_path = "/home/igor/mlprojects/Csgo-NeuralNetwork/output/"
model_save_path = "/home/igor/mlprojects/Csgo-NeuralNetwork/modelsave"
# train_split and test_split 0.1 > x > 0.9 and must add up to 1
train_split = 0.7
val_split = 0.15
test_split = 0.15
num_epochs = 85
batch_size =  3
save = True
##dataset ---------------
dict_override = False
##optimizer -------------
lr = 0.001
momentum = 0.5


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s" % (torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')

n_files = 0
for file in os.listdir(model_save_path):
    n_files += 1
n_files = int(np.ceil((n_files/2)))

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
                return (str(self.folder_system[num_frames]), str(num_frames))
            else:
                idx = idx - num_frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sets path and gets txt/jpg files
        img_path, idx = self.find_folder(idx)
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

    def __init__(self, root_dir, transform=None, dict_override = False):
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

        if dict_override:
            self.folder_system = {2072:'CSGOraw1',
                                2096: 'CSGOraw3',
                                3656: 'CSGOraw4',
                                2500: 'CSGOraw7'}

        else:
            self.folder_system = {}
            for folder in os.listdir(dataset_path):
                folder_len  = 0
                for file in os.listdir(dataset_path + folder):
                    folder_len += 1
                self.folder_system[int((folder_len/2)-1)] = '%s'%(folder)

        for folder_len in self.folder_system:
            self.length += folder_len
        self.length = self.length
        print(self.length)
        
    #returns name of folder that contains specific frame
    def find_folder(self, idx):
        for num_frames in self.folder_system:
            if num_frames >= idx:
                return (str(self.folder_system[num_frames]), str(idx))
            else:
                idx = idx - num_frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #sets path and gets txt/jpg files
        img_path, idx = self.find_folder(idx)
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

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def get_TFNP_classification(outputs, labels):
    TP, TN, FP, FN = 0, 0, 0, 0

    outputs = outputs.detach().cpu()
    labels = labels.detach().cpu().numpy()
    outputs = torch.softmax(outputs, dim=1)
    outputs = np.argmax(outputs, axis=1)
    outputs = outputs.numpy()

    TP = np.logical_and(outputs, labels)
    TN = np.logical_and(np.logical_not(outputs), np.logical_not(labels))
    FP = np.logical_and(outputs, np.logical_not(labels))
    FN = np.logical_and(np.logical_not(outputs), labels)
    return int(TP[0]), int(TN[0]), int(FP[0]), int(FN[0])

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
def train_run(criterion, optimizer, device, train_loader, val_loader=None, save = True):
    if save == False:
        print('ATTENTION: NO PROGRESS WILL BE SAVED\n--------------------------------------')
    tic = time()

    train_inferences = []
    train_losses = []
    train_accs = []
    train_tp, train_tn, train_fp, train_fn = 0, 0, 0, 0

    val_losses = []
    val_accs = []

    print(len(train_loader.dataset))
    log_interval = 10

    num_batches_train = len(train_loader)
    num_batches_val = len(val_loader)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_inference, running_loss, running_acc = 0.0, 0.0, 0.0

        for i, data in enumerate(train_loader):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['label']

            #sends batch to gpu
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            inferecence_start = time()
            outputs = net(inputs)
            inference = time() - inferecence_start
            running_inference += inference

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            acc = binary_acc(outputs, labels)
            running_acc += acc

            TP, TN, FP, FN = get_TFNP_classification(outputs, labels)
            train_tp += TP
            train_tn += TN
            train_fp += FP
            train_fn += FN

            #backprop + optimizer
            loss.backward()
            optimizer.step()

            if (i + 1) % log_interval == 0:  # print every 10 mini-batches
                print('training: [%d, %5d] loss: %.5f acc: %.0f'%\
                    (epoch + 1, i + 1, (running_loss / (i+1)), running_acc / (i+1)))

        train_inferences.append(running_inference / num_batches_train)
        train_losses.append(running_loss / num_batches_train)
        train_accs.append(running_acc / num_batches_train)

        running_val_loss, running_val_acc = 0.0
        #validation run
        for i, data in enumerate(val_loader):
            running_val_loss, running_val_acc = 0.0
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['label']
            #sends batch to gpu
            inputs, labels = inputs.to(device), labels.to(device)
            #run NN
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            acc = binary_acc(outputs, labels)
            running_val_acc += acc
            
            if (i + 1) % log_interval == 0:  # print every 10 mini-batches
                print('validation: [%d, %5d] loss: %.5f acc: %.0f'%\
                    (epoch + 1, i + 1, (running_val_loss / (i+1)), running_val_acc / (i+1)))

        val_losses.append(running_val_loss / num_batches_val)
        val_accs.append(running_val_acc / num_batches_val)


    toc = time()
    toc = toc - tic
    print('Finished Training, elapsed time: %s seconds'%(int(toc)))

    if save == True:
        print('saving state...')
        fname = 'model#%s'%(n_files)
        torch.save(net, fname)
        os.replace(fname, model_save_path+'/'+fname)

        #saving stats
        with open(model_save_path+'/'+fname+'r', 'wb') as file:
            to_dump = {
                'inferences' : train_inferences,
                'losses' : train_losses,
                'accuracy' : train_accs,
                'runtime' : toc,
                'tp':train_tp,
                'tn':train_tn,
                'fp':train_fp,
                'fn':train_fn}
            if val_loader != None:
                to_dump['val_losses'] = val_losses
                to_dump['val_accs'] = val_accs
            pickle.dump(to_dump, file)

    return train_inferences, train_losses, train_accs, val_losses, val_accs

#runs NN in testing mode
def test_run(criterion, device, test_loader, save = True):

    if save == False:
        print('ATTENTION: NO PROGRESS WILL BE SAVED\n--------------------------------------')

    net.eval()
    tic = time()

    test_inferences = []
    test_losses = []
    test_accs = []
    test_tp, test_tn, test_fp, test_fn = 0, 0, 0, 0

    print(len(test_loader.dataset))
    log_interval = 10
    num_batches = len(test_loader)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_inference, running_loss, running_acc = 0.0, 0.0, 0.0

        for i, data in enumerate(test_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['label']
            
            #sends batch to gpu
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass + inference measure
            inferecence_start = time()
            outputs = net(inputs)
            inference = time() - inferecence_start
            running_inference += inference

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            acc = binary_acc(outputs, labels)
            running_acc += acc

            TP, TN, FP, FN = get_TFNP_classification(outputs, labels)
            test_tp += TP
            test_tn += TN
            test_fp += FP
            test_fn += FN

            if (i + 1) % log_interval == 0:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.5f acc: %.0f' %
                      (epoch + 1, i + 1, (running_loss / (i+1)), running_acc / (i+1)))
        
        test_inferences.append(running_inference / num_batches)
        test_losses.append(running_loss / num_batches)
        test_accs.append(running_acc / num_batches)

    toc = time()
    toc = toc - tic
    print('Finished Testing, elapsed time: %s seconds'%(int(toc)))

    if save == True:
        print('saving state...')
        fname = 'model#%s'%(n_files)
        torch.save(net, fname)
        os.replace(fname, model_save_path+'/'+fname)
        with open(model_save_path+'/'+fname+'t', 'wb') as file:
            to_dump = {
                'inferences' : test_inferences,
                'losses' : test_losses,
                'accuracy' : test_accs,
                'runtime' : toc,
                'tp':test_tp,
                'tn':test_tn,
                'fp':test_fp,
                'fn':test_fn}
            pickle.dump(to_dump, file)

    return test_inferences, test_losses, test_accs


net = Net().to(device)

transform = transforms.Compose([
    transforms.Resize([320, 180]),
    # transforms.Resize([57600, 1]),
    transforms.ToTensor(),
])

dataset = CsgoClassificationDataset(dataset_path, transform, dict_override=dict_override)

dataset_len = len(dataset)

train_split = int(np.floor(dataset_len * train_split))
val_split = int(np.floor(dataset_len * val_split))
test_split = int(np.floor(dataset_len * test_split))
while train_split + val_split + test_split != dataset_len:
    train_split += 1
train_set, val_set, test_set = torch.utils.data.random_split( \
    dataset, [train_split, val_split, test_split])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader =   DataLoader(dataset = val_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader =  DataLoader(dataset=test_set,  batch_size=batch_size, shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters())
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)


train_inferences, train_losses, train_accs, val_losses, val_accs = train_run(\
    criterion, optimizer, device, train_loader, val_loader,save=save)

plt.plot(train_inferences)
plt.plot(train_losses)
plt.plot(train_accs)
plt.plot(val_losses)
plt.plot(val_accs)
plt.show()

# test_inferences, test_losses, test_accs = test_run(criterion, device, test_loader, save=save)
# plt.plot(test_inferences)
# plt.plot(test_losses)
# plt.plot(test_accs)
# plt.show()