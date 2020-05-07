from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from random import randint
import pyscreenshot as ImageGrab

#getting first GPU
cuda = torch.device('cuda:0')

#fetching a pre-trained model: maskRCNN_resnet50
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#send model to GPU
model.to(cuda)
#put model in evaluation mode (for testing only)
model.eval()

#coco is the dataset used; these are the names of the categories it recognizes
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


#setting random seed
np.random.seed(42)
#ensure same color is given to each category every time we run the code
COCO_INSTANCE_CATEGORY_COLORS = {}
for i, v in enumerate(COCO_INSTANCE_CATEGORY_NAMES):
    #for each category, there is a list of 3 numbers ranging from 0-255
    COCO_INSTANCE_CATEGORY_COLORS[v] = list(np.random.randint(0, 255, (3, )))

#return a colored mask painting the mask pixels with the corresponding coco category color
def get_coco_category_color_mask(image, coco_category):
    """ 
    image: a binary matrix with 1 where the object is present, and 0 otherwise.
    """
    #create a tensor filled with zeros, with same shape as image
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    #attributes the color values to the corresponding rgb masks for pixels where object is present (image == 1)
    r[image == 1], g[image == 1], b[image == 1] = COCO_INSTANCE_CATEGORY_COLORS[coco_category]
    #stacks red green and blue matrices, creating a tensor along a 3rd axis. 
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def random_colour_masks(image):
  colours = [[0, 255, 0],
             [0, 0, 255],
             [255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask

def detect_on_image(img_path, img_is_path=False, threshold=0.5, rect_th=1, text_th=1, text_size=1):
    """
    img_path: absolute path, or an RGB tensor.
    threshold: determines minimum confidence in order to consider prediction.
    img_is_path: toggles if img_path is an absolute path or an RGB tensor. 
    """
    if img_is_path:
        img = Image.open(img_path).convert("RGB")
    else:
        img = img_path
    img = np.array(img)
    #pointer to transformation function
    #after transforming into pytorch tensor, puts it into composition
    transform = T.Compose([T.ToTensor()])
    #applies transformations, sends iimage to gpu defined on device_'cuda'

    #forward pass, gets network output
    pred = model([transform(img).cuda()])
    #accesses the network prediction scores, detaches it, brings it to CPU and converts it into np array
    pred_scores = list(pred[0]['scores'].detach().cpu().numpy())
    #list of indices of every score above threshold
    pred_t_list = [pred_scores.index(x) for x in pred_scores if x > threshold]
    #index of the worst acceptable prediction score 
    if len(pred_t_list) == 0:
        return None, None
    pred_t = pred_t_list[-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    #gets the coco categories names of labels
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    #list of tuples with x and y coordinates for boxes to be drawn
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    #BUG: what if the worst is the last
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    #RETURNED THIS::: masks, pred_boxes, pred_class, pred_scores[:pred_t+1]
    pred_scores = pred_scores[:pred_t+1]
    # for i in range(len(masks)):
    #     #rgb_mask = random_colour_masks(masks[i])
    #     if len(masks[i].shape) < 2:
    #         continue
    #     rgb_mask = get_coco_category_color_mask(masks[i], pred_class[i])
    #     img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    #     img = cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1], color=(0, 255, 0), thickness=rect_th)
    #     img = cv2.putText(img, f"{pred_class[i]}: {pred_scores[i]:.2f} >= {threshold:.2f}",
    #         pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
    #         text_size, (0, 255, 0), thickness=text_th)
    
    #TODO: enable threshold
    person_pred_boxes = []
    person_pred_scores = []
    for idx, box in enumerate(pred_boxes):
        if pred_class[idx] == 'person': 
            person_pred_boxes.append(box)
            person_pred_scores.append(pred_scores[idx])
    
    
    
    return person_pred_boxes, person_pred_scores

def draw_bbox(img_path, boxes):
    """
    Given four points in an image matrix, draws box onto image and displays.

    img_path: (string) path to image file ==OR== (numpy) RGB Tensor

    boxes: (list) list of tuples (one for each box) containing 4 values eg. (x1, y1, x2, y2) each representing one vertex of the box
    """

    if type(boxes) != list or len(boxes) == 0:
        raise ValueError('no boxes detected')
        
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for box in boxes:
        box_int = [int(i) for i in box]
        img = cv2.rectangle(img, (box_int[0], box_int[1]), (box_int[2], box_int[3]), (0, 255, 0), thickness=1)

    return img



