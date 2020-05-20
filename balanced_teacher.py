import cv2
import os
import numpy as np
import time
import detector
from random import seed
from random import randint

#variable settings
output_path = 'output'  
video_path = "/home/igor/Documents/CSGOraw1.mp4"
samplerate = 1 #video capture framerate
detectionrate = 3 #neural-network passthrogh rate
seed(42)

#setting device etc.
cap = cv2.VideoCapture(video_path)
success = cap.grab() # get the next frame 
absolute_frames = -1
labeled_frames = -1 #starts at -1 so 1st counts as 0th
start_time = time.time()

#storing output preparation
video_title = os.path.basename(video_path)
video_title = str(os.path.splitext(video_title)[0])
output_path_root = output_path + "/" + video_title
os.mkdir(output_path_root)
    
ones_zeros = []

def select_zeros(ones_zeros, random=True):
    one_count = ones_zeros.count(1)
    zeros_idx = []
    
    if random:
        for i in range(one_count):
            idx = randint(0, ones_zeros.len())
            while ones_zeros[j] == 1:
                idx = randint(0, ones_zeros.len())
            zeros_idx.append(idx)
        return  zeros_idx


#while still capturing:
while success:
    absolute_frames += 1
    output_path = output_path_root
    # capture
    _ , img = cap.retrieve()
    success = cap.grab()

    #Output image
    
    if absolute_frames % detectionrate == 0:
        #apply model to image
        pred_boxes = detector.detect_on_image(img, threshold=0.85)
    
    
    if absolute_frames % detectionrate == 0:
        for idx, box in enumerate(pred_boxes):
            if box != None:
                
                box = str(box)
                box = box.replace('[', '')
                box = box.replace(']', '')
                box = box.replace('(', '')
                box = box.replace(')', '')
                box = box.replace(' ', '')

                if idx == 0:
                    labeled_frames += 1
                    frame_title = "%sframe#%s" % (video_title, labeled_frames)
                    output_path = output_path + "/" + frame_title
                    cv2.imwrite(output_path + ".jpg", img)
                    ones_zeros.append(1)

                with open(output_path + ".txt", "w") as file:
                    file.write(box + "\n")

            else:
                ones_zeros.append(0)

    end_time = time.time()
    avg_time_frame = absolute_frames / (end_time - start_time)
    print(avg_time_frame)

    cv2.imshow("screen", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

zeros_idx = select_zeros(ones_zeros)

for zero in zeros_idx:
    labeled_frames += 1
    cap.set(1, zero)
    _, img = cap.read()
    
    frame_title = "%sframe#%s" % (video_title, labeled_frames)
    output_path = output_path + "/" + frame_title
    cv2.imwrite(output_path + ".jpg", img)

    with open(output_path + ".txt", "w") as file:
        file.write("")
