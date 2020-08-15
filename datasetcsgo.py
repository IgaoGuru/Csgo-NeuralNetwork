import json
from torch.utils.data import Dataset
import os
from PIL import Image
import torch

img_size = (1280, 720)

def load_dict_dataset(dataset_root_path, dataset_name):
    """

    :param dataset_root_path: root path to all datasets
    :param dataset_name: name of the dataset
    :return: a dictionary whose key is the frame name and value is a pair (class, bboxes)
    """


    with open(f"{dataset_root_path}{dataset_name}.json") as f:
        data = json.load(f)

    dict_dataset = {}
    clazzes = []
    for frame_name, v in data['_via_img_metadata'].items():
        bboxes = []
        classes = []
        if len(v['regions']):
            #print(frame_name)
            for vr in v['regions']:
                try:
                    clazz = vr['region_attributes']['class']
                except:
                    clazz = vr['region_attributes']['CSGo']
                bbox = vr['shape_attributes']['x'],\
                       vr['shape_attributes']['y'],\
                       vr['shape_attributes']['x'] + vr['shape_attributes']['width'],\
                       vr['shape_attributes']['y'] + vr['shape_attributes']['height']
                classes.append(clazz)
                bboxes.append(bbox)
        dict_dataset[frame_name.split('.')[0]] = (classes, bboxes)
    return dict_dataset

class CsgoDataset(Dataset):
    """ Dataset for CSGo Bounding Box detection of people """

    ignore_empty_bboxes = True
    KNOWN_CLASSES = [
        "BackGround",
        "Terrorist",
        "CounterTerrorist"
    ]

    def __init__(self, root_path, classes=None, transform=None, scale_factor=None):
        self.root_path = root_path
        self.transform = transform
        self.scale_factor = scale_factor
        if classes is None:
            self.classes = self.KNOWN_CLASSES
        else:
            self.classes = classes
        self.classes = ["Background"] + self.classes
        dict_datasets = {}
        for root, dirs, files in os.walk(root_path):
            for dir in dirs:
                dict_datasets[dir] = load_dict_dataset(root_path, dir)
            break
        if len(dict_datasets) == 0:
            raise Exception('No dataset folder was found!')
        self.dict_frames = {}
        for dir, dict_dataset in dict_datasets.items():
            for k, v in dict_dataset.items():
                if self.ignore_empty_bboxes and len(v[1]) > 0:
                    if set(v[0]).issubset(set(self.classes)):
                        frame_key = dir + "/" + k
                        self.dict_frames[frame_key] = v
        self.length = len(self.dict_frames)
        self.frame_keys = list(self.dict_frames.keys())

    def __len__(self):
        return self.length

    def __str__(self):
        return self.root_path + " " + str(self.length)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        if self.transform:
            img = self.transform(img)
        #clazz = self.dict_frames[self.frame_keys[idx]][0]

        bboxes = torch.tensor(self.dict_frames[self.frame_keys[idx]][1], dtype=torch.float)
        if self.scale_factor != None:
            bboxes = bboxes * self.scale_factor
        labels = torch.tensor([self.classes.index(c) for c in self.dict_frames[self.frame_keys[idx]][0]], dtype=torch.int64)
        return img, bboxes, labels

    def get_image_path(self, idx):
        return self.root_path + self.frame_keys[idx] + '.jpg'

    def get_image(self, idx):
        img_path = self.get_image_path(idx)
        img = Image.open(img_path)
        return img

    def split(self, train, val, seed=None):
        if seed:
            torch.random.manual_seed(seed)
        train_size = int(train * len(self))
        val_size = int(val * len(self))
        test_size = len(self) - (train_size + val_size)
        return torch.utils.data.random_split(self, [train_size, val_size, test_size])