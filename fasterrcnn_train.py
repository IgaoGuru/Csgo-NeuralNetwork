from torch.utils.data.dataloader import DataLoader
from shutil import move
from torchvision import transforms
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from os import walk
import torchvision
import cv2
import numpy as np
import torch.nn as nn
# Our own libraries
import fastercnn
import datasetcsgo
from util import loss_dict_template

print(f"torch version: {torch.__version__}")
print(f"Torch CUDA version: {torch.version.cuda}")
print(f"torchvision version: {torchvision.__version__}")
print(f"opencv version: {cv2.__version__}")

print("")

SEED = 42
torch.manual_seed(SEED)
train_only = 'tr'  
scale_factor = 0.2
num_epochs = 200
checkpoints = [0, 19, 49, 79, 99, 119, 149, 179, 199] #all epoch indexes where the network should be saved
model_number = 999

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')

# dataset_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetworkPaulo\\data\\datasets\\"  #remember to put "/" at the end
dataset_path = "/home/igor/mlprojects/Csgo-NeuralNetworkold/data/datasets/"  #remember to put "/" at the end
model_save_path = '/home/igor/mlprojects/modelsave/'

model_save_path = f"{model_save_path}model#{model_number}"  

#net_func = fastrcnn.get_custom_fasterrcnn
net_func = fastercnn.get_fasterrcnn_small

if train_only == 'ct':
    classes = ["CounterTerrorist"]
if train_only == 'tr':
    classes = ["Terrorist"]
else:
    classes = ["Terrorist", "CounterTerrorist"]

model = net_func(num_classes=len(classes)+1, num_convs_backbone=1, num_backbone_out_channels=16)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

#model.apply(init_weights)
model = model.to(device)
print(model)

if scale_factor != None:
    transform = transforms.Compose([
    transforms.Resize([int(720*scale_factor), int(1280*scale_factor)]),
    transforms.ToTensor(), # will put the image range between 0 and 1
])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

#dataset = CsgoPersonFastRCNNDataset(dataset_path, transform)
dataset = datasetcsgo.CsgoDataset(dataset_path, classes=classes, transform=transform, scale_factor=scale_factor)

# a simple custom collate function, just to show the idea
def my_collate(batch):
    imgs = [item[0] for item in batch]
    targets = [(item[1][0], item[1][1]) for item in batch]
    img_paths = [item[2] for item in batch]
    return [imgs, targets, img_paths]

def my_collate_2(batch):
    imgs = [item[0] for item in batch]
    bboxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return [imgs, bboxes, labels]

batch_size = 1

train_set, _, _ = dataset.split(train=0.2, val=0.8, seed=SEED)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)

optimizer = optim.Adam(model.parameters())
losses = []
log_interval = len(train_loader) // 1
min_avg_loss = 1e6

print(f"Started training! Go have a coffee/mate/glass of water...")
print(f"Log interval: {log_interval}")
print(f"Please wait for first logging of the training")

loss_total_dict = loss_dict_template
for epoch in range(num_epochs):  # loop over the dataset multiple times

    loss_per_epoch = loss_dict_template
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        imgs, bboxes, labels = data
        images = list(im.to(device) for im in imgs)

        targets = [{'boxes': b.to(device), 'labels': l.to(device)} for b, l in zip(bboxes, labels)]
        optimizer.zero_grad()

        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())
        loss_value = loss.item()
        
        loss_per_epoch['loss_sum'].append(loss_value)
        loss_per_epoch['loss_classifier'].append(loss_dict['loss_classifier'].item())
        loss_per_epoch['loss_box_reg'].append(loss_dict['loss_box_reg'].item())
        loss_per_epoch['loss_objectness'].append(loss_dict['loss_objectness'].item())
        loss_per_epoch['loss_rpn_box_reg'].append(loss_dict['loss_rpn_box_reg'].item())
        print(loss_per_epoch)

        if (i + 1) % log_interval == 0:
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / log_interval))
            avg_loss = running_loss / log_interval
            losses.append(avg_loss)
            running_loss = 0.0
            print([(k, v.item()) for k, v in loss_dict.items()])

            if epoch in checkpoints: 
                min_avg_loss = avg_loss
                print(f"Saving net at: {model_save_path}")
                torch.save(model.state_dict(), model_save_path + 'e' + f'{epoch}' + '.th')

        loss.backward()

        optimizer.step()

#print(f"Saving net at: {model.__class__.__name__ + '.th'}")
#torch.save(model.state_dict(), model.__class__.__name__ + ".th")

plt.plot(losses[1:])
plt.show()

