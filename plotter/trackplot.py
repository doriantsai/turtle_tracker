import cv2
import os
import glob
import code
from PIL import Image
import numpy as np
from classifier.Classifier import Classifier
##############################################################################################
White = (250, 250, 250)
Blue = (57,127,252)
Purple = (198,115,255)
Green = (0,200,120)
Black = (0,0,0)
Red = (250,0,0)
font = cv2.FONT_HERSHEY_SIMPLEX
sf = 0.3



def track2box(textfile,img,line_thickness):
    '''
    Given a img and corrisponding textfile with track detections with lines being new tracks and
    each line having class, x1,y1,x2,y2, confidence in detections and track id
    Create a box around each tracked turtle
    '''
    imgw, imgh = img.shape[1], img.shape[0]
    datalines = []
    with open(textfile) as f:
        for line in f:
            cls,b,c,d,e,conf,id = line.split()
            x1 = round(float(b)*imgw)
            x2 = round(float(d)*imgw)
            y1 = round(float(c)*imgh)
            y2 = round(float(e)*imgh)
            dataline = [int(cls), x1,y1,x2,y2,float(conf),int(id)]
            datalines.append(dataline)

    return datalines

def boxwithid(datalines,
                      img, 
                      line_thickness=2):
        '''
        Datalines = [class, x1,y1,x2,y2, confidence, track id, classifier class, conf class]
        Crete a box with specific string and colour.
        '''
        for dataline in datalines:
            x1, y1, x2, y2 = dataline[1:5]
            conf, id, cls, conf2 = dataline[5], dataline[6], dataline[7], dataline[8]
            #change results depending on class
            if cls == 0: colour = Green #normal turtle = 0
            elif cls == 1: colour = Blue #painted turtle = 1
            else: colour = Black #something weird is happening
            
            conf_str = format(conf*100.0, '.0f')
            conf_str2 = format(conf2*100.0, '.0f')
            detect_str = '{} D:{} C:{}'.format(id, conf_str, conf_str2)
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                    colour, line_thickness) #box around tutle
            boxwithtext(img, int(x1), int(y1), detect_str, colour, line_thickness)

def boxwithtext(img, x1, y1, text, colour, thickness):
        '''
        Given a img, starting x,y cords, text and colour, create a filled in box of specified colour
        and write the text
        '''
        font_scale = max(1,0.000005*max(img.shape))
        p = 5 #padding
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x1-p, y1-p), (x1+text_size[0]+p, y1-text_size[1]-(2*p)), colour, -1)
        cv2.putText(img, text, (x1,y1-10), font, font_scale, White, thickness)

base_data_path = '/home/raineai/Turtles/datasets/trim_vid'
data_location = os.path.join(base_data_path, 'output/041219-0569AMsouth_trim')
video_in = os.path.join(base_data_path,'041219-0569AMsouth_trim.mp4')
video_out = os.path.join(base_data_path,'041219-0569AMsouth_trim.mp4').split('.')[0]+'-5frames.mp4'

clasifier = Classifier()
txtlist = sorted(glob.glob(os.path.join(data_location, '*.txt')))
imglist = sorted(glob.glob(os.path.join(data_location, '*.jpg')))

# Video setup
vidcap = cv2.VideoCapture(video_in)
total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),30,(w,h),isColor=True)

transformed_imglist = []
# for every tracked image:
for i, imgname in enumerate(imglist):
        img = cv2.imread(imgname)
        imgw, imgh = img.shape[1], img.shape[0]

        datalines = track2box(txtlist[i], img, 1) #fetch class and xyxy details
        for dataline in datalines: #for every turtule
                cls_img = clasifier.read_image(imgname)
                cls_img_crop = clasifier.crop_image(cls_img,dataline[1:5],1,1) #crop turtle
                pred_class, predictions = clasifier.classify_image(cls_img_crop) #classifiy it
                #add classified details to datalist
                dataline.append(int(pred_class[0]))
                dataline.append(1-predictions[pred_class].item()) 
        #create boxes aroud turtles
        boxwithid(datalines,img)
        #img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
        #cv2.imshow('images', img)
        #cv2.waitKey(0)
        transformed_imglist.append(img) 
        print(f'{i+1}/{len(imglist)}: image {imgname.replace(data_location,"")} processed')

#make video
for image in transformed_imglist:
    out.write(image)

out.release()

code.interact(local=dict(globals(), **locals()))

####################################################################3
