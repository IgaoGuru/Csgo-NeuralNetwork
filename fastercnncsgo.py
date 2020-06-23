import cv2
import numpy as np
import time
import torch
import torchvision.transforms as T
import fastercnn

def count_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.grab():
        frame_idx += 1
    return frame_idx


torch.manual_seed(0)
if torch.cuda.is_available():
    cuda = torch.device("cuda:0")
    print("Running on: %s"%(torch.cuda.get_device_name(cuda)))
else:
    cuda = torch.device("cpu")
    print('running on: CPU')

#fetching a pre-trained model: maskRCNN_resnet50
#net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#net = fastrcnn.get_fastrcnn_mobile()
net = torch.load("FasterRCNN.th")
#send model to GPU
net.to(cuda)
#put model in evaluation mode (for testing only)
net.eval()

video_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetwork\\CSGOraw2.mp4"
sample_rate = 10 # video capture framerate
neuralnet_detection_rate = 1 # neural-network passthrogh rate
avg_time_frame = 0
neuralnet_threshold = 0.53

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
        bboxes, pred_cls, pred_scores = get_prediction_fastrcnn(img, net, neuralnet_threshold, img_is_path=False)
    else:
        bboxes = None
    # write current masks and bboxes on image
    if bboxes is not None:
        for i in range(len(bboxes)):
            print(bboxes[i][0])

            img = cv2.rectangle(img, bboxes[i][0], bboxes[i][1], color=(0, 255, 0), thickness=rect_th)
            img = cv2.putText(img, f"{pred_cls[i]}: {pred_scores[i]:.2f} >= {neuralnet_threshold:.2f}",
                              bboxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                              text_size, (0, 255, 0), thickness=text_th)

    # show images
    cv2.imshow('img', img)
    time.sleep(0.2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    show_detection = True






