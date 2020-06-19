import cv2
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as T

def count_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.grab():
        frame_idx += 1
    return frame_idx

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

COCO_INSTANCE_CATEGORY_COLORS = {}
for i, v in enumerate(COCO_INSTANCE_CATEGORY_NAMES):
    COCO_INSTANCE_CATEGORY_COLORS[v] = list(np.random.randint(0, 255, (3, )))

def get_coco_category_color_mask(image, coco_category):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = COCO_INSTANCE_CATEGORY_COLORS[coco_category]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img_path, model, threshold, img_is_path=True):
    if img_is_path:
        img = Image.open(img_path).convert("RGB")
    else:
        img = img_path
    img = np.array(img)
    transform = T.Compose([T.ToTensor()])
    img = transform(img).cuda()
    pred = model([img])
    pred_scores = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t_list = [pred_scores.index(x) for x in pred_scores if x > threshold]
    if len(pred_t_list) == 0:
        return None, None, None, None
    pred_t = pred_t_list[-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class, pred_scores[:pred_t+1]

def detect_on_img(img, net, threshold, text_size=1, text_th=1, rect_th=1):
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masks, boxes, pred_cls, pred_scores = get_prediction(img, net, threshold, img_is_path=False)
    if masks is None:
        return img
    for i in range(len(masks)):
        # rgb_mask = random_colour_masks(masks[i])
        if len(masks[i].shape) < 2:
            continue
        rgb_mask = get_coco_category_color_mask(masks[i], pred_cls[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        img = cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        #bbox_text = f"{pred_cls[i]}: {pred_scores[i]:.2f} >= {threshold:.2f}"
        bbox_text = f"{pred_cls[i]}: {pred_scores[i]:.2f}"
        img = cv2.putText(img, bbox_text,
                          boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                          text_size, (0, 255, 0), thickness=text_th)
    return img


#getting first GPU
cuda = torch.device('cuda:0')

#fetching a pre-trained model: maskRCNN_resnet50
net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#send model to GPU
net.to(cuda)
#put model in evaluation mode (for testing only)
net.eval()

video_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetwork\\CSGOraw2.mp4"
sample_rate = 10 # video capture framerate
neuralnet_detection_rate = 1 # neural-network passthrogh rate
avg_time_frame = 0
neuralnet_threshold = 0.7

#print("Counting number of frames (this might take a while)...")
#num_frames = count_video_frames(video_path)
#print(f"Number of frames: {num_frames}")

# grab first frame
cap = cv2.VideoCapture(video_path)
success = cap.grab() # get the next frame
frame_idx = -1
# while there are frames to grab
cap = cv2.VideoCapture(video_path)
show_detection = False
masks = None
bboxes = []
pred_cls = []
pred_scores = []
text_size = 1
text_th = 2
rect_th = 1
while cap.grab():
    frame_idx += 1
    # Skip frame according to frame sample rate
    if not (frame_idx + 1) % sample_rate == 0:
        continue
    # grab image
    _, img = cap.retrieve()
    # Skip frame according to neuralnet detection rate
    if (frame_idx + 1) % neuralnet_detection_rate == 0:
        masks, bboxes, pred_cls, pred_scores = get_prediction(img, net, neuralnet_threshold, img_is_path=False)
    else:
        masks = None
    # write current masks and bboxes on image
    if masks is not None:
        for i in range(len(masks)):
            # rgb_mask = random_colour_masks(masks[i])
            if len(masks[i].shape) < 2:
                continue
            rgb_mask = get_coco_category_color_mask(masks[i], pred_cls[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            img = cv2.rectangle(img, bboxes[i][0], bboxes[i][1], color=(0, 255, 0), thickness=rect_th)
            img = cv2.putText(img, f"{pred_cls[i]}: {pred_scores[i]:.2f} >= {neuralnet_threshold:.2f}",
                              bboxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                              text_size, (0, 255, 0), thickness=text_th)

    # show images
    cv2.imshow('img', img)
    #time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    show_detection = True






