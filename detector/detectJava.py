'''
from the image specified, shows the ground truth detection and yolov5 algorithm detection
'''
import torch
import cv2
import numpy as np
import argparse
import glob
import os

#set up
basepath = os.path.join('/home/raineai/Turtles')
yolopath = os.path.join(basepath,'yolov5')
weights = os.path.join(basepath,'yolov5/runs/train/exp55/weights/best.pt')
imgpath = os.path.join(basepath,'datasets/yolov5-small/train')
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
    savename = imgname+'.txt'
    with open(savepath, 'w') as f:
        for p in predarray:
            x1,y1,x2,y2 = p[0:4].tolist()
            conf, cls = p[4], int(p[5])
            f.write(f'{x1:.6f} {x2:.6f} {y1:.6f} {y2:.6f} {conf:.6f} {cls:g}\n')
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
        x1, y1, x2, y2 = p[0:4].tolist()
        conf, cls = p[4], int(p[5])
        #change results depending on class
        if cls == 0: colour, text = Green, 'Turtle' #normal turtle = 0
        elif cls == 1: colour, text = Purple, 'Pained Turtle' #painted turtle = 1
        else: colour, text = Black, 'Unknown class number' #something weird is happening
        
        conf_str = format(conf*100.0, '.0f')
        detect_str = '{}: {}'.format(text, conf_str)
        
        #plotting
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                colour, line_thickness) #box around tutle
        boxwithtext(img, int(x1), int(y1), detect_str, colour, line_thickness)
        
def boxwithtext(img, x1, y1, text, colour, thickness):
    '''
    Given a img, starting x,y cords, text and colour, create a filled in box of spcified colour
    and write the text
    '''
    font_scale = 0.5
    p = 5 #padding
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x1-p, y1-p), (x1+text_size[0]+p, y1-text_size[1]-(2*p)), colour, -1)
    cv2.putText(img, text, (x1,y1-10), font, font_scale, White, thickness)


def singleimg(imgname,groundtruth,savetext):
    imglocation = imgpath+'/images/'+imgname+'.jpg'
    #savepath = [some location]

    img = cv2.imread(imglocation)
    
    if ~groundtruth:
        textfile = imgpath+'/labels/'+imgname+'.txt'
        text2box(textfile,img,Red,1)
    
    ### Load Model ###
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.hub.load(yolopath, 'custom', path=weights, source='local')
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


def multipleimg(imglist,sourceimg):
    ''' In Progress
    Given a list of images, make predition boxes for each class on each image
    '''
    #set up
    line_thickness = 1
    no_t = 0
    array = [] 

    ### Load Model ### 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.hub.load(yolopath, 'custom', path=weights, source='local')
    model = model.to(device) 
    #work through each img
    for i, imgname in enumerate(imglist):
        img = cv2.imread(imgname)

        ### Ground Truth ###
        short_name = imgname.replace(sourceimg, '')
        text_name = short_name.replace('.jpg','.txt')
        textfile = imgpath+'/labels/'+text_name
        text2box(textfile,img, Red, line_thickness)
            
        ### Detection ###
        results = model([img], size=img_size)
        predarray = pred2array(results)    
        predarray2box(predarray,img,line_thickness+1)
            
        #show image
        sf = 0.7
        img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
        cv2.imshow('images', img)
        cv2.waitKey(0)
            #cv2.destroyAllWindows()

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
    sourceimage = os.path.join(basepath, 'datasets/yolov5-small/train/images')
    imglist = glob.glob(os.path.join(sourceimage, '*.jpg'))
    multipleimg(imglist,sourceimage)

if __name__ == "__main__":
    #opt = parse_opt()
    opt = 0
    main(opt)
    #singleimg('TS071217-00002-090_jpg.rf.a96fa7935a2bf3fc31e8bdbdd82fc427', True, False)


