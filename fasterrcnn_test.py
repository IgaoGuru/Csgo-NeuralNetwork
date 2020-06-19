import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import cv2
import time
# Our own libraries
import fastercnn
import datasetcsgo


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


acc = 0.0
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
        bboxes, pred_cls, pred_scores = fastercnn.get_prediction_fastercnn(
            imgs[0].to(device), net, threshold, category_names=categories, img_is_path=False)
        cls_gt = np.array(categories)[[t.item() for t in targets[0]]]
        print(cls_gt, bboxes_gt[0])
        end = time.time()
        net_elapseds.append(end - start)
        avg_net_elapsed = np.mean(net_elapseds)
        #print(f"Avg inference time: {avg_net_elapsed}")
    else:
        bboxes = None
        pred_cls = None
        pred_scores = None

    # write current masks and bboxes on image
    if bboxes is not None:
        for i in range(len(bboxes)):

            img = cv2.rectangle(img, bboxes[i][0], bboxes[i][1], color=(0, 255, 0), thickness=rect_th)
            img = cv2.putText(img, f"{pred_cls[i]}: {pred_scores[i]:.2f}",
                              bboxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                              text_size, (0, 255, 0), thickness=text_th)

    cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
