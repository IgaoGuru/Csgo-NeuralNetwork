import cv2
import os
import numpy as np
import time
import detector
from random import seed
from random import randint

#variable settings
output_path = 'output'  
video_path = "/home/igor/Documents/csgofootage/CSGOraw7.mp4"
samplerate = 1 #video capture framerate
detectionrate = 2 #neural-network passthrogh rate
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

def select_zeros(ones_zeros, absolute_frames, random=True):
    one_count = ones_zeros.count(1)
    zeros_idx = []
    if one_count == 0:
        print("no entities  were detected in the footage")
    
    if random:
        for i in range(one_count):
            idx = randint(0, len(ones_zeros)-1)
            while ones_zeros[idx] == 1 or ones_zeros[idx] == 2 or idx > absolute_frames:
                idx = randint(0, len(ones_zeros)-1)
            zeros_idx.append(idx)
        return  zeros_idx

#NOTE:cap.set(1, 11021)

# while still capturing:
while success:
    absolute_frames += 1
    output_path = output_path_root
    # capture
    _ , img = cap.retrieve()
    success = cap.grab()

    #Output image
    
    if absolute_frames % detectionrate == 0:
        #apply model to image
        try:
            pred_boxes = detector.detect_on_image(img, threshold=0.89)
        except RuntimeError:
            #if memory runs out, as if frame was not detected
            ones_zeros.append(2)
            continue
    
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
    else:
        #filling in the gaps of all frames
        #that werent analized
        ones_zeros.append(2)

    end_time = time.time()
    avg_time_frame = absolute_frames / (end_time - start_time)
    print(avg_time_frame)

    cv2.imshow("screen", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cv2.destroyAllWindows()

# with open(output_path_root + ".txt", "w") as file:
#     file.write("%s"%(ones_zeros))
# ones_zeros = np.genfromtxt("/home/igor/mlprojects/Csgo-NeuralNetwork/output/CSGOraw2.txt", delimiter=",", dtype = int)
# ones_zeros = list(ones_zeros)

zeros_idx = select_zeros(ones_zeros, absolute_frames)

for zero in zeros_idx:
    output_path = output_path_root
    labeled_frames += 1
    cap.set(1, zero)
    _, img = cap.read()
    
    frame_title = "%sframe#%s" % (video_title, labeled_frames)
    output_path = output_path + "/" + frame_title
    cv2.imwrite(output_path + ".jpg", img)

    with open(output_path + ".txt", "w") as file:
        file.write("")
