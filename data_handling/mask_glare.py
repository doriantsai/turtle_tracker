import cv2
import numpy as np

video_file = '041219-0569AMsouth.MP4'
frame_no = 300
GET_HSV_VALS = 1



vidcap = cv2.VideoCapture(video_file)
success, image = vidcap.read()
count = 0

while success:
    if count == frame_no:
       break
    count += 1
    success, image = vidcap.read()

if GET_HSV_VALS:
    def callback(x):
        global H_low,H_high,S_low,S_high,V_low,V_high,K
        #assign trackbar position value to H,S,V High and low variable
        H_low = cv2.getTrackbarPos('low H','controls')
        H_high = cv2.getTrackbarPos('high H','controls')
        S_low = cv2.getTrackbarPos('low S','controls')
        S_high = cv2.getTrackbarPos('high S','controls')
        V_low = cv2.getTrackbarPos('low V','controls')
        V_high = cv2.getTrackbarPos('high V','controls')
        K = cv2.getTrackbarPos('K','controls')



    cv2.namedWindow('controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('controls',550, 20)

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('mask',960, 640)

    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('res',960, 640)

    cv2.namedWindow('morph', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('morph',960, 640)

    H_low, S_low, V_low, K = 0,0,0,1
    H_high, S_high, V_high = 179,255,255


    cv2.createTrackbar('low H','controls',0,179,callback)
    cv2.createTrackbar('high H','controls',179,179,callback)

    cv2.createTrackbar('low S','controls',0,255,callback)
    cv2.createTrackbar('high S','controls',255,255,callback)

    cv2.createTrackbar('low V','controls',0,255,callback)
    cv2.createTrackbar('high V','controls',255,255,callback)

    cv2.createTrackbar('K','controls',1,200,callback)

    while True:

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # print(hsv.shape)
        hsv_low = np.array([H_low, S_low, V_low], np.uint8)
        hsv_high = np.array([H_high, S_high, V_high], np.uint8)

        #making mask for hsv range
        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        #masking HSV value selected color becomes black
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(K+2,K+2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(K,K))
        erode = cv2.erode(~mask, erode_kernel, iterations=2)
        dilate = cv2.dilate(erode, kernel, iterations=1)
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
        res = cv2.bitwise_and(image, image, mask=~dilate)
        
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        cv2.imshow('morph', dilate)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

if not GET_HSV_VALS:
    [H_low, S_low, V_low],[H_high, S_high, V_high]=[0, 7, 0], [179, 255, 255]

print([H_low, S_low, V_low],[H_high, S_high, V_high],K)
