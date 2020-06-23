from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import cv2
import datasetcsgo
import numpy as np
import torch.tensor
from time import sleep

dataset_path = '/home/igor/mlprojects/Csgo-NeuralNetwork/data/datasets/'

transform = transforms.Compose([
    #transforms.Resize([200, 200]),
    transforms.ToTensor(),
])

classes = ["Terrorist", "CounterTerrorist"]

dataset = datasetcsgo.CsgoDataset(dataset_path, classes=classes, transform=transform)
_, val_set, test_set = dataset.split(train=0.7, val=0.15, seed=42)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

sample_rate = 1
frame_idx = -1
text_size = 1
text_th=1
rect_th=1

categories = ["Background"] + classes

for i, data in enumerate(test_loader):
    frame_idx += 1
    # Skip frame according to frame sample rate
    if not (frame_idx + 1) % sample_rate == 0:
        continue

    imgs, bboxes_gt, targets = data
    cls_gt = np.array(categories)[[t.item() for t in targets[0]]]

    img = imgs[0].numpy().copy().transpose(1, 2, 0)

    bboxes_gt = bboxes_gt[0].int()

    # write current masks and bboxes on image
    if bboxes_gt is not None:
        for b in range(np.shape(bboxes_gt)[0]):
            # print('LENGTH::::: %s'%(i))

            pt1 = bboxes_gt[b][0].item(), bboxes_gt[b][1].item()
            pt2 = bboxes_gt[b][2].item(), bboxes_gt[b][3].item()

            img = cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
            img = cv2.putText(img, cls_gt[b],
                              pt1, cv2.FONT_HERSHEY_SIMPLEX,
                              text_size, (0, 255, 0), thickness=text_th)

    cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

