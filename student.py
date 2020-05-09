#TODO: use only one (RGB) channel
import numpy as np
import pandas as pd
import os
from torch.utils import data
import torch
from torchvision import transforms
from natsort import natsorted, ns
import cv2
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = "/home/igor/mlprojects/Csgo-NeuralNetwork/output/"
# NOTE: possibly remove in next version??
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
                label = torch.as_tensor(label, dtype=torch.int16)
                label_shape = np.shape(label)
            #if file is blank (no data in the image), create -1 matrix
            else:
                label = torch.zeros([1, 5], dtype=torch.int16)
                label[label==0] = -1
            
            sample = {'image':img, 'label':label}
            
        #apply transforms
        #TODO: farofa aqui hein
        if self.transform:
            img = self.transform(sample['image'])
            sample['image'] = img

        return sample

transform = transforms.Compose([
    transforms.Resize([320, 180]),
    transforms.ToTensor()
])

dataset = CsgoPersonDataset(dataset_path, transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=5,
                                        shuffle=True, num_workers=4)

