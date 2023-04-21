import cv2
import numpy as np
import time
import os.path

video_in = '041219-0569AMsouth.MP4'

frames_out = 'frames/'+video_in.split('.')[0]+'/'

if not os.path.isdir(frames_out):
    os.mkdir(frames_out)
else:

    vidcap = cv2.VideoCapture(video_in)
    time.sleep(3)
    background=0
    SHOW = 0
    success, image = vidcap.read()
    count = 0
    
    while success:
        if count >= 0 and count <= 30:
            file_name = frames_out+'frame_'+str(count).zfill(6)+'.jpg'
            cv2.imwrite(file_name, image)
        
        success, image = vidcap.read()
        count = count+1
        print(count)

        #if count > 100:
        #    break
      
    vidcap.release()
#[0, 44, 0] [179, 255, 255] 6

