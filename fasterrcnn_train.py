from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision
import cv2
# Our own libraries
import fastercnn
import numpy as np
import datasetcsgo
import torch.nn as nn

print(f"torch version: {torch.__version__}")
print(f"Torch CUDA version: {torch.version.cuda}")
print(f"torchvision version: {torchvision.__version__}")
print(f"opencv version: {cv2.__version__}")

print("")

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print('running on: CPU')

dataset_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetworkPaulo\\data\\datasets\\"  #remember to put "/" at the end

#net_func = fastrcnn.get_custom_fasterrcnn
net_func = fastercnn.get_fasterrcnn_small

classes = ["Terrorist", "CounterTerrorist"]
model = net_func(num_classes=len(classes)+1, num_convs_backbone=5, num_backbone_out_channels=64)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

model.apply(init_weights)
model = model.to(device)


transform = transforms.Compose([
    #transforms.Resize([200, 200]),
    transforms.ToTensor(), # will put the image range between 0 and 1
])

#dataset = CsgoPersonFastRCNNDataset(dataset_path, transform)
dataset = datasetcsgo.CsgoDataset(dataset_path, classes=classes, transform=transform)

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

batch_size = 4

train_set, _, _ = dataset.split(train=0.7, val=0.15, seed=42)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_2)

optimizer = optim.Adam(model.parameters())
num_epochs = 100
losses = []
log_interval = len(train_loader) // 1
min_avg_loss = 1e6

print(f"Started training! Go have a coffee/mate/glass of water...")
print(f"Log interval: {log_interval}")
print(f"Please wait for first logging of the training")

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        imgs, bboxes, labels = data
        images = list(im.to(device) for im in imgs)

        targets = [{'boxes': b.to(device), 'labels': l.to(device)} for b, l in zip(bboxes, labels)]

        optimizer.zero_grad()
        
        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())
        loss_value = loss.item()

        running_loss += loss_value

        if (i + 1) % log_interval == 0:
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / log_interval))
            avg_loss = running_loss / log_interval
            losses.append(avg_loss)
            running_loss = 0.0
            print([(k, v.item()) for k, v in loss_dict.items()])
            if avg_loss < min_avg_loss:
                min_avg_loss = avg_loss
                print(f"Saving net at: {net_func.__name__ + '.th'}")
                torch.save(model.state_dict(), net_func.__name__ + ".th")

        loss.backward()

        optimizer.step()


#print(f"Saving net at: {model.__class__.__name__ + '.th'}")
#torch.save(model.state_dict(), model.__class__.__name__ + ".th")

plt.plot(losses[1:])
plt.show()

