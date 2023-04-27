import torch
import cv2
import numpy as np
import argparse
import glob
import os

import logging
# Add this line before the PyTorch log messages appear This will suppress the
# logging messages from PyTorch in the 'yolov5' logger.
logging.getLogger('yolov5').setLevel(logging.CRITICAL)

import code
'''
from the image specified, shows the ground truth detection and yolov5 algorithm detection
'''

#set up
basepath = os.path.join('/home/dorian/Code/turtles')
yolopath = os.path.join(basepath,'yolov5_turtles')
weights = os.path.join(basepath,'yolov5_turtles/runs/train/exp7/weights/last.pt')
imgpath = os.path.join(basepath, 'turtle_datasets/job10_041219-0-1000/split_data/train/images')
labelpath = os.path.join(basepath, 'turtle_datasets/job10_041219-0-1000/split_data/train/labels')
savepath_default = os.path.join(imgpath, '../detections')
os.makedirs(savepath_default, exist_ok=True)

IMGsuffix = '.PNG'
font = cv2.FONT_HERSHEY_SIMPLEX
img_size = 1280
White = (250, 250, 250)
Blue1 = (252,127,57)
Purple = (255,115,198)
Green = (120,200,0)
Black = (0,0,0)
Red = (0,0,250)



def pred2array(results):
    '''
    takes a model predictions and converts them to an array
    array = xmin, xmax, ymin, ymax, confidence, class
    '''
    pred = results.pred[0]
    predarray = []
    for i in range(len(pred)):
        row = []
        for j in range(6):
            row.append(pred[i,j].item())
        predarray += (row)
    predarray = np.array(predarray)
    predarray = predarray.reshape(len(pred),6)
    return predarray


def savepredarray(predarray,savepath): #still to test
    '''
    save the predition results in a textformat ready for recovery
    '''
    # savename = imgname+'.txt'
    with open(savepath, 'w') as f:
        for p in predarray:
            x1,y1,x2,y2 = p[0:4].tolist()
            conf, cls = p[4], int(p[5])
            #f.write(f'{x1:.6f} {x2:.6f} {y1:.6f} {y2:.6f} {conf:.6f} {cls:g}\n')
    return True


def text2box(textfile,img,colour,line_thickness):
    '''
    takes a groundtruth text file with xy cords and plots a boxes
    on the linked image in specified colour
    '''
    
    imgw, imgh = img.shape[1], img.shape[0]
    x1,y1,x2,y2,i = [],[],[],[],0
    with open(textfile) as f:
        for line in f:
            a,b,c,d,e = line.split()
            w = round(float(b)*imgw)
            h = round(float(c)*imgh)
            y1.append(h+round(float(e)*imgh/2))
            y2.append(h-round(float(d)*imgh/2))
            x1.append(w+round(float(d)*imgw/2))
            x2.append(w-round(float(d)*imgw/2))
            cv2.rectangle(img, (x1[i], y1[i]), (x2[i], y2[i]), colour, line_thickness)
            i += 1


def predarray2box(predarray,img, line_thickness):
    '''
    from a prediction array draws boxes around the object as well as labeling
    the boxes on the linked in specified colour
    '''
    for p in predarray:
        # if i>4:
        #     break
        x1, y1, x2, y2 = p[0:4].tolist()
        conf, cls = p[4], int(p[5])
        #change results depending on class
        if cls == 0: colour, text = Green, 'Turtle' #normal turtle = 0
        elif cls == 1: colour, text = Purple, 'Pained Turtle' #painted turtle = 1
        else: colour, text = Black, 'Unknown class number' #something weird is happening
        
        conf_str = format(conf*100.0, '.0f')
        detect_str = '{}: {}'.format(text, conf_str)
        
        #print(x1)
        # i += 1
        #plotting
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                colour, line_thickness) #box around tutle
        boxwithtext(img, int(x1), int(y1), detect_str, colour, line_thickness)
     
        
def boxwithtext(img, x1, y1, text, colour, thickness):
    '''
    Given a img, starting x,y cords, text and colour, create a filled in box of specified colour
    and write the text
    '''
    font_scale = 0.5 # TODO make these functions of image size
    p = 5 #padding
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x1-p, y1-p), (x1+text_size[0]+p, y1-text_size[1]-(2*p)), colour, -1)
    cv2.putText(img, text, (x1,y1-10), font, font_scale, White, thickness)


def singleimg(imgname,groundtruth,savetext):
    imglocation = imgpath+'/images/'+imgname+IMGsuffix 
    #savepath = [some location]

    img = cv2.imread(imglocation)
    
    if ~groundtruth:
        textfile = imgpath+'/labels/'+imgname+'.txt'
        text2box(textfile,img,Red,1)
    
    ### Load Model ###
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'weights file = {weights}')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, trust_repo=True)
    model = model.to(device) #load model

    ### Detection ###
    results = model([img], size=img_size)
    predarray = pred2array(results)    
    predarray2box(predarray,img,1)
    if savetext: v=1 #savepredarray(predarray,savepath) #still to do
        
    #show image
    sf = 0.7
    img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
    cv2.imshow('images', img)
    cv2.waitKey(0)


def multipleimg(imgdir_path: str,
                groundtruth_path: str,
                savepath: str = savepath_default,
                SHOW: bool = False, 
                SAVE: bool =True):
    ''' In Progress
    Given a list of images, make predition boxes for each class on each image
    '''
    #set up
    line_thickness = 1
    no_t = 0
    array = [] 
    sf = 0.4

    ### Load Model ### 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = torch.hub.load(yolopath, 'custom', path=weights, source='local')
    print(f'weights file = {weights}')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, trust_repo=True)
    model = model.to(device) 
    model.eval() # model into evaluation mode
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = True
    model.max_det = 1000
    
    
    # get groundtruth text list 
    txtlist = sorted(glob.glob(os.path.join(groundtruth_path, '*.txt')))
    # apply sorted to ensure consistent indexing between txtlist (labels) and
    # imglist (images)
    
    # get image list
    imglist = sorted(glob.glob(os.path.join(imgdir_path, '*.PNG')))
    
    #iterate through each img
    maximg = 1
    for i, imgname in enumerate(imglist):
        
        if i>maximg:
            break
        
        print(f'{i+1}/{len(imglist)}: {os.path.basename(imgname)}')
        img = cv2.imread(imgname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        
        # short_name = imgname.replace(sourceimg, '')
        # text_name = short_name.replace(IMGsuffix, '.txt')
        # textfile = imgpath+'/labels'+text_name
        
            
        ### Detection ###
        results = model([img], size=img_size)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ### Ground Truth ###
        # code.interact(local=dict(globals(), **locals()))
        text2box(txtlist[i],img, Red, line_thickness)
        
        predarray = pred2array(results)
        print(predarray[0:4]*sf)    
        print(results.pandas().xyxy[0])
        predarray2box(predarray,img,line_thickness+1)
        
        code.interact(local=dict(globals(), **locals()))
        
        
        #show image
        if SHOW:
            img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
            cv2.imshow('images', img)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
        
        # code.interact(local=dict(globals(), **locals()))
        # save image
        if SAVE:
            
            cv2.imwrite(os.path.join(savepath, os.path.basename(imgname)), img)
            
        #count
       # print(len(predarray))
       # no_t = no_t + len(predarray)
       # array.append(len(predarray))
    
   # print(no_t)
   # print(array)
   # print(no_t/1087.0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgname', default='', type=str, help='image to process')
    parser.add_argument('-g', '--groundtruth', action='store_true', 
            help='are ground truth boxes wanted, default=True, write to change default')
    parser.add_argument('-s', '--savetext', action='store_true', 
            help='if the detection wants to be saved, default=True, write to change default')
    opt = parser.parse_args()
    return opt


def main(opt):
    #imgname = 'TS071217-00002-090_jpg.rf.a96fa7935a2bf3fc31e8bdbdd82fc427'
    #singleimg(**vars(opt))
    #######Still to do########
    #how to make it work for a folder of images?
    # sourceimage = os.path.join(basepath, 'datasets/job12/train/images')
    
    multipleimg(imgpath,labelpath)


if __name__ == "__main__":
    #opt = parse_opt()
    opt = 0
    main(opt)
    #singleimg('TS071217-00002-090_jpg.rf.a96fa7935a2bf3fc31e8bdbdd82fc427', True, False)


