import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import cv2
import time
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
# Our own libraries
import fastercnn
import datasetcsgo

IMG_SHAPE = (720, 1280)

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')

dataset_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetworkPaulo\\data\\datasets\\"

transform = transforms.Compose([
    #transforms.Resize([200, 200]),
    transforms.ToTensor(),
])

classes = ["Terrorist", "CounterTerrorist"]

dataset = datasetcsgo.CsgoDataset(dataset_path, classes=classes, transform=transform)
_, val_set, test_set = dataset.split(train=0.7, val=0.15, seed=42)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

#net_func = fastrcnn.get_custom_fasterrcnn
net_func = fastercnn.get_fasterrcnn_mobile


net = net_func(num_classes=len(classes)+1)
print(f"Loading net from: {net_func.__name__ + '.th'}")
net.load_state_dict(torch.load(net_func.__name__ + ".th"))

net.eval()
net.to(device)

threshold = 0.7
sample_rate = 1
neuralnet_detection_rate = 1
frame_idx = -1
text_size = 1
text_th=1
rect_th=1

net_elapseds = []
categories = ["Background"] + classes


def get_pred_error(bboxes_gt, bboxes_pred):
    np_bboxes_gt = bboxes_gt.cpu().numpy()[0]
    #print(np_bboxes_gt.shape)
    #print(np_bboxes_gt)
    if bboxes_pred is None:
        #print(max(IMG_SHAPE), np_bboxes_gt.shape[0])
        return max(IMG_SHAPE) * np_bboxes_gt.shape[0]
    np_bboxes_pred = np.array(bboxes_pred).reshape(-1, 4)
    #print(np_bboxes_pred.shape)
    #print(np_bboxes_pred)
    dist_mtx_1 = cdist(np_bboxes_gt[:, 0:2], np_bboxes_pred[:, 0:2], metric="euclidean")
    dist_mtx_2 = cdist(np_bboxes_gt[:, 2:], np_bboxes_pred[:, 2:], metric="euclidean")
    bboxes_pred_error_mtx = dist_mtx_1 + dist_mtx_2
    #print(bboxes_pred_error_mtx)
    bboxes_pred_error = int(sum(np.min(bboxes_pred_error_mtx, axis=0)))
    return bboxes_pred_error

acc = 0.0
bboxes_pred_errors = []
for i, data in enumerate(test_loader):
    frame_idx += 1
    # Skip frame according to frame sample rate
    if not (frame_idx + 1) % sample_rate == 0:
        continue

    imgs, bboxes_gt, targets = data
    img = imgs[0].numpy().copy().transpose(1, 2, 0)

    # Skip frame according to neuralnet detection rate
    if (frame_idx + 1) % neuralnet_detection_rate == 0:
        start = time.time()
        bboxes_pred, pred_cls, pred_scores = fastercnn.get_prediction_fastercnn(
            imgs[0].to(device), net, threshold, category_names=categories, img_is_path=False)
        cls_gt = np.array(categories)[[t.item() for t in targets[0]]]
        #print(frame_idx)
        #print(cls_gt)
        #print(pred_cls)
        end = time.time()
        net_elapseds.append(end - start)
        avg_net_elapsed = np.mean(net_elapseds)
        #print(f"Avg inference time: {avg_net_elapsed}")
    else:
        bboxes_pred = None
        pred_cls = None
        pred_scores = None

    bboxes_pred_error = get_pred_error(bboxes_gt, bboxes_pred)
    bboxes_pred_errors.append(bboxes_pred_error)

    img = cv2.UMat(img).get() # this solves weird cv2.rectangle error
    if bboxes_pred is not None:
        for b in range(len(bboxes_pred)):
            pt1 = int(bboxes_pred[b][0][0]), int(bboxes_pred[b][0][1])
            pt2 = int(bboxes_pred[b][1][0]), int(bboxes_pred[b][1][1])
            img = cv2.rectangle(img, pt1, pt2, (0, 255, 0), rect_th)
            img = cv2.putText(img, f"{pred_cls[b]}: {pred_scores[b]:.2f}",
                              pt1, cv2.FONT_HERSHEY_SIMPLEX,
                              text_size, (0, 255, 0), thickness=text_th)
    cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    time.sleep(0.1)

    cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #time.sleep(1)
    #print("-----------------------------------------------------------------------")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

median_bboxes_pred_error = int(np.median(bboxes_pred_errors))
mean_bboxes_pred_error = int(np.mean(bboxes_pred_errors))
stddev_bboxes_pred_error = int(np.median(bboxes_pred_errors))

print(f"Median bboxes pred error: {median_bboxes_pred_error}")
print(f"Mean (+-stddev) bboxes pred error: {mean_bboxes_pred_error} +- {stddev_bboxes_pred_error}")

hist, bins = np.histogram(bboxes_pred_errors)
hist = hist / np.sum(hist)
plt.bar(bins[:-1], hist, width=np.diff(bins))

stddev_left = max(0, mean_bboxes_pred_error - stddev_bboxes_pred_error)
stddev_right = mean_bboxes_pred_error + stddev_bboxes_pred_error
stddev_width = stddev_right - stddev_left
plt.bar(mean_bboxes_pred_error, np.max(hist), width=stddev_width, color="green")

plt.vlines(median_bboxes_pred_error, 0, np.max(hist), color="orange", label='Median')
plt.vlines(mean_bboxes_pred_error, 0, np.max(hist), color="green", label='Mean (+- stddev)')
plt.legend()
plt.show()
