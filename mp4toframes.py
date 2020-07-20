import cv2
import os
import numpy

datasets_folder = "/home/igor/Documents/footageimages/"
videos_folder = "/home/igor/Documents/csgofootage/"

video_name = "CSGOigor1"
video_ext = ".mp4"

try:
    # os.mkdir(f"{root_path}{datasets_folder}")
    os.mkdir(f"{datasets_folder}{video_name}")
except:
    pass

cap = cv2.VideoCapture(f"{videos_folder}{video_name}{video_ext}")
success = cap.grab() # get the next frame
frame_rate = 60
frame_idx = -1
while success:
    frame_idx += 1
    _ , img = cap.retrieve()
    cap.set(1, frame_rate * frame_idx)
    success = cap.grab()
    print(frame_idx)
    try:
        # cv2.imshow("screen", img)
        img_path = f"{datasets_folder}{video_name}/{video_name}_{frame_idx}.jpg"
        cv2.imwrite(img_path, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        continue

cv2.destroyAllWindows()
