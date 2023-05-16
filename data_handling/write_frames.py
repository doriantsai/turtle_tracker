"""write_frames.py

script to take video and split it into frames into a specified folder
"""

import cv2 as cv
import numpy as np
import time
import os

# video input in
video_dir = '/home/dorian/Code/turtles/turtle_datasets'
video_in = '081220-0310AMsouth-ME2D.mp4'
video_path = os.path.join(video_dir, video_in)

# output frames
video_name = video_in.split('.')[0]
frame_dir = os.path.join(video_dir, video_name)

os.makedirs(frame_dir, exist_ok=True)
# frames_out = 'frames/'+video_in.split('.')[0]+'/'

vidcap = cv.VideoCapture(video_path)
time.sleep(3)
background=0
SHOW = 0
success, image = vidcap.read()
if success:
    print(f'Starting video read: {video_in}')
    
count = 0
MAX_COUNT = 1001
while success:
    # if count >= 0 and count <= 30:
    file_name = os.path.join(frame_dir, video_name + '_frame_' + str(count).zfill(6)+'.jpg')
    print(f'frames: {count}: {file_name}')
        
    cv.imwrite(file_name, image)
    
    success, image = vidcap.read()
    count += 1
    
    if count >= MAX_COUNT:
        print(f'Stop write_frames. Count = {count}, MAX_COUNT ({MAX_COUNT}) reached.')
        break

print('done')
vidcap.release()
#[0, 44, 0] [179, 255, 255] 6

