#TODO: use only one (RGB) channel
import numpy as np
import pandas as pd
import os
from torch.utils import data
import torch
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
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.length = 0
        #dictionary that marks what the last frame of each folder is
        #ie. number of examples in specific folder
        self.folder_system = {5098:'CSGOraw1'}

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
            if num_frames >= idx:
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
            #if file has important data
            if os.stat(label_path).st_size != 0:
                label = np.genfromtxt(file, delimiter=',')
                if np.shape(label) == (5,):
                    label = np.reshape(label, (1, 5))
                label = torch.as_tensor(label, dtype=torch.int16)
            #if file is blank (no data in the image)
            else:
                label = torch.zeros([1, 5], dtype=torch.int16)
                label[label==0] = -1
        return {'image':img, 'label':label}

    def draw_bbox_batch(batch):
        """
        Show image with boxes for a batch of images.
        """
        images_batch, label_batch = \
                batch['image'], batch['label']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        grid_border_size = 2

        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

        # for box in boxes:
        #     box_int = [int(i) for i in box]
        #     img = cv2.rectangle(img, (box_int[0], box_int[1]), (box_int[2], box_int[3]), (0, 255, 0), thickness=1)

        # return img

dataset = CsgoPersonDataset(dataset_path)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=5,
                                        shuffle=True, num_workers=4)


# for i in range(1, 1000):
#     labelss = dataset[i]['label']
#     print(labelss)


for i in range(1, 150):
    dataiter = iter(dataloader)
    data = dataiter.next()
    img, labels = data['image'], data['label']
    print(labels)
    print(labels.size())

# for data in dataloader:
#     print(data)

# CsgoPersonDataset.draw_bbox_batch(dataloader)