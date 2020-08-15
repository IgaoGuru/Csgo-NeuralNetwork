import numpy as np
import torch
from os import walk
import pickle
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import cv2
import time
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
# Our own libraries
import fastercnn
import stat_interpreter_reg as stats
import datasetcsgo

IMG_SHAPE = (1080, 1920)

SEED = 42
torch.manual_seed(SEED)
test_only = 'tr'
scale_factor = 1 #Leave 1 if no scaling should be done

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')

# dataset_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetworkPaulo\\data\\datasets\\"  #remember to put "/" at the end
dataset_path = "/home/igor/mlprojects/Csgo-NeuralNetworkold/data/datasets/"  #remember to put "/" at the end
results_folder = '/home/igor/Documents/csgotesting/' 
model_path = '/home/igor/mlprojects/modelsave/model#29e49'
# model_path = '/home/igor/mlprojects/modelsave/model#12e49'

transform = transforms.Compose([
    transforms.Resize([int(1080*scale_factor), int(1920*scale_factor)]),
    transforms.ToTensor(), # will put the image range between 0 and 1
])

if test_only == 'ct':
    classes = ["CounterTerrorist"]
if test_only == 'tr':
    classes = ["Terrorist"]
else:
    classes = ["Terrorist", "CounterTerrorist"]

dataset = datasetcsgo.CsgoDataset(dataset_path, classes=classes, transform=transform, scale_factor=scale_factor)

train_set, val_set, test_set = dataset.split(train=0.7, val=0.15, seed=SEED)
test_loader = DataLoader(val_set, batch_size=1, shuffle=False)

#net_func = fastrcnn.get_custom_fasterrcnn
#net_func = fastercnn.get_fasterrcnn_mobile
net_func = fastercnn.get_fasterrcnn_small


net = net_func(num_classes=len(classes)+1, num_convs_backbone=5, num_backbone_out_channels=64)
print(f"Loading net from: {model_path + '.th'}")
net.load_state_dict(torch.load(model_path + ".th"))

net.eval()
net.to(device)

threshold = 0.1
sample_rate = 1
neuralnet_detection_rate = 1
frame_idx = -1
text_size = 1
text_th=1
rect_th=1

net_elapseds = []
categories = ["Background"] + classes

def get_pred_error(bboxes_gt, bboxes_pred):
    #calculates diagonal of image rectangle
    MAX_ERROR = int(np.sqrt(IMG_SHAPE[0]**2 + IMG_SHAPE[1]**2))
    print("-----------------------------------------------------------")
    #transform bboxes_gt into np.array(n, 4) 
    np_bboxes_gt = bboxes_gt.cpu().numpy()[0]
    print(f"Ground truth bboxes shape: {np_bboxes_gt.shape}")
    #if nothing is predicted, penalize with maximum possible error per gt_bbox
    if bboxes_pred is None:
        print(f"Error for no prediction: {MAX_ERROR * np_bboxes_gt.shape[0]}")
        return MAX_ERROR * np_bboxes_gt.shape[0]
    #transform bboxes_pred into np.array(m, 4)
    np_bboxes_pred = np.array(bboxes_pred).reshape(-1, 4)
    print(f"Pred bboxes shape: {np_bboxes_pred.shape}")
    #calculates distance matrix(n, m) between ground truth and pred.
    dist_mtx_1 = cdist(np_bboxes_gt[:, 0:2], np_bboxes_pred[:, 0:2], metric="euclidean")
    dist_mtx_2 = cdist(np_bboxes_gt[:, 2:], np_bboxes_pred[:, 2:], metric="euclidean")
    bboxes_pred_error_mtx = (dist_mtx_1 + dist_mtx_2) / 2
    print(f"Error mtx shape: {bboxes_pred_error_mtx.shape}")
    print(bboxes_pred_error_mtx)
    print(np.min(bboxes_pred_error_mtx, axis=0))
    bboxes_pred_error = int(sum(np.min(bboxes_pred_error_mtx, axis=0)) / bboxes_pred_error_mtx.shape[1])
    num_lacking_bboxes = abs(np_bboxes_pred.shape[0] - np_bboxes_gt.shape[0])
    print(f"num_lacking_bboxes: {num_lacking_bboxes}")
    bboxes_pred_error += num_lacking_bboxes * MAX_ERROR
    print("-----------------------------------------------------------")
    return bboxes_pred_error

# def get_pred_error(bboxes_gt, bboxes_pred):
#     np_bboxes_gt = bboxes_gt.cpu().numpy()[0]
#     #print(np_bboxes_gt.shape)
#     #print(np_bboxes_gt)
#     if bboxes_pred is None:
#         #print(max(IMG_SHAPE), np_bboxes_gt.shape[0])
#         return max(IMG_SHAPE) * np_bboxes_gt.shape[0]
#     np_bboxes_pred = np.array(bboxes_pred).reshape(-1, 4)
#     #print(np_bboxes_pred.shape)
#     #print(np_bboxes_pred)
#     dist_mtx_1 = cdist(np_bboxes_gt[:, 0:2], np_bboxes_pred[:, 0:2], metric="euclidean")
#     dist_mtx_2 = cdist(np_bboxes_gt[:, 2:], np_bboxes_pred[:, 2:], metric="euclidean")
#     bboxes_pred_error_mtx = dist_mtx_1 + dist_mtx_2
#     #print(bboxes_pred_error_mtx)
#     bboxes_pred_error = int(sum(np.min(bboxes_pred_error_mtx, axis=0)))
#     return bboxes_pred_error

acc = 0.0
bboxes_pred_errors = []
loss_total = {}
loss_total_dict = {
    'loss_sum' : [],
    'loss_classifier' : [],
    'loss_box_reg' : [],
    'loss_objectness' : [],
    'loss_rpn_box_reg' : []
}
for i, data in enumerate(test_loader):
    frame_idx += 1
    # Skip frame according to frame sample rate
    if not (frame_idx + 1) % sample_rate == 0:
        continue

    imgs, bboxes_gt, targets = data
    print(bboxes_gt)
    img = imgs[0].numpy().copy().transpose(1, 2, 0)

    # Skip frame according to neuralnet detection rate
    if (frame_idx + 1) % neuralnet_detection_rate == 0:

        #compute losses with marotagem
        net.train()

        images = list(im.to(device) for im in imgs)
        targets_ = [{'boxes': b.to(device), 'labels': l.to(device)} for b, l in zip(bboxes_gt, targets)]

        loss_dict = net(images, targets_)
        loss = sum(l for l in loss_dict.values())
        loss_value = loss.item()

        loss_total_dict['loss_sum'].append(loss_value)
        loss_total_dict['loss_classifier'].append(loss_dict['loss_classifier'].item())
        loss_total_dict['loss_box_reg'].append(loss_dict['loss_box_reg'].item())
        loss_total_dict['loss_objectness'].append(loss_dict['loss_objectness'].item())
        loss_total_dict['loss_rpn_box_reg'].append(loss_dict['loss_rpn_box_reg'].item())

        net.eval()

        # update loss_total
        if i == 0:
            for loss_type in loss_dict:
                loss_total[loss_type] = []
        for loss_type in loss_dict:
            loss_total[loss_type].append(loss_dict[loss_type].item())

        #forward prop
        start = time.time()
        bboxes_pred, pred_cls, pred_scores = fastercnn.get_prediction_fastercnn(
            imgs[0].to(device), net, threshold, category_names=categories, img_is_path=False)
        cls_gt = np.array(categories)[[t.item() for t in targets[0]]]

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

    if len(bboxes_gt[0]) != 1:
        for b in range(len(bboxes_gt)):
            pt1 = (int(bboxes_gt[0][b][0]), int(bboxes_gt[0][b][1]))
            pt2 = (int(bboxes_gt[0][b][2]), int(bboxes_gt[0][b][3]))
            img = cv2.rectangle(img, pt1, pt2, (255 , 0, 0), rect_th)
    else:
        pt1 = (int(bboxes_gt[0][0][0]), int(bboxes_gt[0][0][1]))
        pt2 = (int(bboxes_gt[0][0][2]), int(bboxes_gt[0][0][3]))
        img = cv2.rectangle(img, pt1, pt2, (255 , 0, 0), rect_th)

    cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    time.sleep(1.2)
    #print("-----------------------------------------------------------------------")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#save loss_results:
result_index = 0
for i in walk(results_folder):
    result_index+=1
with open(f"{results_folder}test-results#{result_index}", 'wb') as file:
    pickle.dump(loss_total_dict, file) 
cv2.destroyAllWindows()

median_bboxes_pred_error = int(np.median(bboxes_pred_errors))
mean_bboxes_pred_error = int(np.mean(bboxes_pred_errors))
stddev_bboxes_pred_error = int(np.median(bboxes_pred_errors))

print(f"Median bboxes pred error: {median_bboxes_pred_error}")
print(f"Mean (+-stddev) bboxes pred error: {mean_bboxes_pred_error} +- {stddev_bboxes_pred_error}")

# hist, bins = np.histogram(bboxes_pred_errors)
# hist = hist / np.sum(hist)
# plt.bar(bins[:-1], hist, width=np.diff(bins))

# stddev_left = max(0, mean_bboxes_pred_error - stddev_bboxes_pred_error)
# stddev_right = mean_bboxes_pred_error + stddev_bboxes_pred_error
# stddev_width = stddev_right - stddev_left
# plt.bar(mean_bboxes_pred_error, np.max(hist), width=stddev_width, color="green")

# plt.vlines(median_bboxes_pred_error, 0, np.max(hist), color="orange", label='Median')
# plt.vlines(mean_bboxes_pred_error, 0, np.max(hist), color="green", label='Mean (+- stddev)')
# plt.legend()

stats.interpreter(loss_dict=loss_total_dict, mode=2)
stats.stdev_mean(loss_dict=loss_total_dict)
plt.show()
