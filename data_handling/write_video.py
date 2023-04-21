import cv2
import numpy as np
import time
import os.path

video_in = '041219-0569AMsouth.MP4'

video_out = 'MASKED/'+video_in.split('.')[0]+'-mask2000-3000.mp4'
if not os.path.isfile(video_out):

    vidcap = cv2.VideoCapture(video_in)
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),30,(w,h),isColor=True)
    time.sleep(3)
    background=0
    SHOW = 0
    success, image = vidcap.read()
    count = 0
    hsv_low = np.array([0,95,0], np.uint8)
    hsv_high = np.array([179,255,255],np.uint8)
    K = 25
    while success:
        if count >= 2000 and count <=3000:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_low, hsv_high)
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(K+2,K+2))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(K,K))
            erode = cv2.erode(~mask, erode_kernel, iterations=3)
            dilate = cv2.dilate(erode, kernel, iterations=1)

            res = cv2.bitwise_and(image, image, mask=~dilate)
            if SHOW:
                cv2.namedWindow('res', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('res', 960,640)
                cv2.imshow('res', res)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    break
                elif k == 32:
                    cv2.destroyAllWindows()
            
            out.write(res)
       
            success,image = vidcap.read()
            count = count+1
            print(str(count)+'/'+str(total))
        else:
            success,image = vidcap.read()
            count = count+1

            if count >3001:
                break
        #if count > 100:
        #    break
      
    out.release()
    vidcap.release()
#[0, 44, 0] [179, 255, 255] 6

