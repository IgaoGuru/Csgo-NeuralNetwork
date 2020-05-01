import cv2
import os
import numpy as np
import time
import detector

#variable settings
output_path = 'output'  
video_path = "/home/igor/Documents/CSGOraw1.mp4"
samplerate = 1 #video capture framerate
detectionrate = 3 #neural-network passthrogh rate
avg_time_frame = 0

#setting device etc.
cap = cv2.VideoCapture(video_path)
success = cap.grab() # get the next frame
num_frames = 0
absolute_frames = 0
start_time = time.time()

#storing output preparation
video_title = os.path.basename(video_path)
video_title = str(os.path.splitext(video_title)[0])
output_path_root = output_path + "/" + video_title
os.mkdir(output_path_root)

#while still capturing:
while success:
    absolute_frames += 1
    output_path = output_path_root
    #framecount and capture
    num_frames += samplerate
    if num_frames % samplerate == 0:
        _ , img = cap.retrieve()
    success = cap.grab()

    #Output image
    frame_title = "%sframe#%s" % (video_title, num_frames)
    output_path = output_path + "/" + frame_title
    cv2.imwrite(output_path + ".jpg", img)
    
    if num_frames % detectionrate == 0:
        #apply model to image
        pred_boxes, pred_scores = detector.detect_on_image(img)
    
    with open(output_path + ".txt", "w") as file:
        if num_frames % detectionrate == 0 and pred_boxes != None:
            for idx, box in enumerate(pred_boxes):
                box = str(box)
                box = box.replace('[', '')
                box = box.replace(']', '')
                box = box.replace('(', '')
                box = box.replace(')', '')
                box = box.replace(' ', '')

                confidence = str(pred_scores[idx])
                confidence = confidence.replace('[', '')
                confidence = confidence.replace(']', '')
                confidence = confidence.replace('(', '')
                confidence = confidence.replace(')', '')
                confidence = confidence.replace(' ', '')
                file.write(confidence + ',' + box + "\n")
        pass
    end_time = time.time()
    avg_time_frame = absolute_frames / (end_time - start_time)
    print(avg_time_frame)

    cv2.imshow("screen", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
