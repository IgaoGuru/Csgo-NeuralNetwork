#TODO: use only one (RGB) channel
import numpy as np
import pandas as pd
import os
from torch.utils import data
import torch
from natsort import natsorted, ns
import cv2
from PIL import Image

dataset_path = "/home/igor/mlprojects/Csgo-NeuralNetwork/output/"
video_folder_name = "CSGOraw1"
label_index = 12
label_path = dataset_path + video_folder_name +'/'+ ("%sframe#%s" % (video_folder_name, label_index)) + '.txt'
bboxes = pd.read_csv(label_path)
# print(bboxes)

class CsgoPersonDataset(data.Dataset):
    """preety description."""

    length = -1

    def __init__(self, root_dir, transform=No1ne):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.length = 0
        #dictionary that marks what the last frame of each folder is
        #ie. number of examples in specific folder
        self.folder_system = {3000:'CSGOraw1'}

        for idx, (folder, _, _) in enumerate(os.walk(self.root_dir)):
            if idx == 0:
                continue
            sorted_files = natsorted(os.listdir(folder), alg=ns.IGNORECASE)
            for file in sorted_files:
                if file[-3:] == 'txt':
                    self.length += 1
        
        #returns name of folder that contains specific frame
    def find_folder(self, idx):
        for num_frames in self.folder_system:
            if num_frames > idx:
                return str(self.folder_system[num_frames])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.find_folder(idx)
        img_name = "%sframe#%s" % (img_path, idx)
        img_path = os.path.join(self.root_dir,
                                img_path, img_name)
        img_path_ext = img_path + '.jpg'
        img = Image.open((img_path_ext))
        img = np.array(img)
        label_path = str(img_path) + '.txt'
        with open(label_path) as file:
            if os.stat(label_path).st_size != 0:
                label = np.genfromtxt(file, delimiter=',')
            else:
                label = "label file empty!"
        return img, label

dataset = CsgoPersonDataset(dataset_path)
