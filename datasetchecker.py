from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import cv2
import datasetcsgo
import numpy as np
import torch.tensor
from time import sleep

dataset_path = '/home/igor/mlprojects/Csgo-NeuralNetworkold/data/datasets/'

transform = transforms.Compose([
    #transforms.Resize([200, 200]),
    transforms.ToTensor(),
])

classes = ["Terrorist", "CounterTerrorist"]

dataset = datasetcsgo.CsgoDataset(dataset_path, classes=classes, transform=transform, scale_factor=1)
train_set, val_set, test_set = dataset.split(train=0.7, val=0.15, seed=42)
test_loader = DataLoader(train_set, batch_size=1, shuffle=False)

sample_rate = 1
frame_idx = -1
text_size = 1
text_th=1
rect_th=1

ct = 0
tr = 0
both = 0
empty = 0

categories = ["Background"] + classes

for i, data in enumerate(test_loader):
    frame_idx += 1
    # Skip frame according to frame sample rate
    if not (frame_idx + 1) % sample_rate == 0:
        continue

    img, bboxes_gt, targets = data
    cls_gt = np.array(categories)[[t.item() for t in targets[0]]]

    img = img[0].numpy().copy().transpose(1, 2, 0)

    img = cv2.UMat(img).get() # this solves weird cv2.rectangle error

    # write current masks and bboxes on image
    if len(bboxes_gt[0]) != 0:
        ct_lock = 0
        tr_lock = 0
        if len(bboxes_gt[0]) != 1:
            for b in range(len(bboxes_gt[0])):
                pt1 = (int(bboxes_gt[0][b][0]), int(bboxes_gt[0][b][1]))
                pt2 = (int(bboxes_gt[0][b][2]), int(bboxes_gt[0][b][3]))
                img = cv2.rectangle(img, pt1, pt2, (255 , 0, 0), rect_th)
                if cls_gt[b] == 'Terrorist':
                    if tr_lock == 1:
                        continue
                    tr+=1
                    tr_lock+=1 
                if cls_gt[b] == 'CounterTerrorist':
                    if ct_lock == 1:
                        continue
                    ct+=1
                    ct_lock+=1
        else:
            pt1 = (int(bboxes_gt[0][0][0]), int(bboxes_gt[0][0][1]))
            pt2 = (int(bboxes_gt[0][0][2]), int(bboxes_gt[0][0][3]))
            img = cv2.rectangle(img, pt1, pt2, (255 , 0, 0), rect_th)
            if cls_gt[0] == 'Terrorist':
                tr+=1
            if cls_gt[0] == 'CounterTerrorist':
                ct+=1
    else:
        empty += 1

    cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # sleep(2.5)

    # print(ct)
    # print(tr)
    # print(both)
    # print(empty)
    # print('aaaaaa \n')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

