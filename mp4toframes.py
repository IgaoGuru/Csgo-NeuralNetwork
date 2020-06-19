import cv2
import os

root_path = "C:\\Users\\User\\Documents\\GitHub\\Csgo-NeuralNetworkPaulo\\data\\"
datasets_folder = "datasets\\"
videos_folder = "videos\\"

video_name = "CSGOCounterTerrorist1"
video_ext = ".mp4"

try:
    os.mkdir(f"{root_path}{datasets_folder}
    os.mkdir(f"{root_path}{datasets_folder}{video_name}")
except:
    pass

cap = cv2.VideoCapture(f"{root_path}{videos_folder}{video_name}{video_ext}")
success = cap.grab() # get the next frame
frame_rate = 100
frame_idx = -1
while success:
    frame_idx += 1
    _ , img = cap.retrieve()
    success = cap.grab()
    if not frame_idx % frame_rate == 0:
        continue
    try:
        cv2.imshow("screen", img)
        img_path = f"{root_path}{datasets_folder}{video_name}\\{video_name}_{frame_idx}.jpg"
        cv2.imwrite(img_path, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        continue

cv2.destroyAllWindows()
