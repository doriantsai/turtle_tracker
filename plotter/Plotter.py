#! /usr/bin/env python3

"""Plotter.py
A collection of common plotting functions used by the Detector, Tracker and Classifier
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class Plotter:
    
    # RGB
    White = (250, 250, 250)
    Blue = (57,127,252)
    Purple = (198,115,255)
    Green = (0,200,120)
    Black = (0,0,0)
    Red = (250,0,0)
    
    def __init__(self, 
                 width: int, 
                 height: int):
        self.image_width = width
        self.image_height = height
        
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.img_size = 1280
        
    
    def groundtruth2box(self, 
                 textfile,
                 image,
                 colour=Red,
                 line_thickness=3):
        '''
        takes a groundtruth text file with xy cords and plots a boxes
        on the linked image in specified colour
        '''
        
        imgw, imgh = image.shape[1], image.shape[0]
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
                cv.rectangle(image, (x1[i], y1[i]), (x2[i], y2[i]), colour, line_thickness)
                i += 1


    def predarray2box(self, 
                      predarray,
                      img, 
                      line_thickness=2):
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
            if cls == 0: colour, text = self.Green, 'Turtle' #normal turtle = 0
            elif cls == 1: colour, text = self.Purple, 'Pained Turtle' #painted turtle = 1
            else: colour, text = self.Black, 'Unknown class number' #something weird is happening
            
            conf_str = format(conf*100.0, '.0f')
            detect_str = '{}: {}'.format(text, conf_str)
            
            #print(x1)
            # i += 1
            #plotting
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                    colour, line_thickness) #box around tutle
            self.boxwithtext(img, int(x1), int(y1), detect_str, colour, line_thickness)
            
            
    def boxwithtext(self, img, x1, y1, text, colour, thickness):
        '''
        Given a img, starting x,y cords, text and colour, create a filled in box of specified colour
        and write the text
        '''
        #font_scale = 0.5 # 
        font_scale = max(1,0.000005*max(img.shape))
        p = 5 #padding
        text_size, _ = cv.getTextSize(text, self.font, font_scale, thickness)
        cv.rectangle(img, (x1-p, y1-p), (x1+text_size[0]+p, y1-text_size[1]-(2*p)), colour, -1)
        cv.putText(img, text, (x1,y1-10), self.font, font_scale, self.White, thickness)
        

    def boxwithid(self, 
                  datalines,
                    img, 
                    line_thickness=2):
        '''
        Datalines = [class, x1,y1,x2,y2, confidence, track id, classifier class, conf class]
        Crete a box with specific string and colour.
        NOTE box coordinates come in as normalised!
        '''
        for dataline in datalines:
            x1n, y1n, x2n, y2n = dataline[1:5]
            conf, id, cls, conf2 = dataline[5], dataline[6], dataline[7], dataline[8]
            #change results depending on class
            if cls == 0: colour = self.Green #normal turtle = 0
            elif cls == 1: colour = self.Blue #painted turtle = 1
            else: colour = self.Black #something weird is happening
            
            # import code
            # code.interact(local=dict(globals(), **locals()))
            conf_str = format(conf*100.0, '.0f')
            conf_str2 = format(conf2*100.0, '.0f')
            detect_str = '{} D:{} C:{}'.format(id, conf_str, conf_str2)
            
            x1 = x1n * float(self.image_width)
            x2 = x2n * float(self.image_width)
            y1 = y1n * float(self.image_height)
            y2 = y2n * float(self.image_height)
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                    colour, line_thickness) #box around turtle
            self.boxwithtext(img, int(x1), int(y1), detect_str, colour, line_thickness)


    def track2box(self, textfile):
        '''
        Given a img and corrisponding textfile with track detections with lines being new tracks and
        each line having class, x1,y1,x2,y2, confidence in detections and track id
        Returns a array with nessary information to make a box.
        '''
        imgw, imgh = self.image_width, self.image_height
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

    def save_image(self, image, save_path, color_format='RGB'):
        """save_image

        Args:
            image (_type_): _description_
            save_path (_type_): _description_
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if color_format == 'RGB':
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(save_path, image)


if __name__ == "__main__":
    
    print('Plotter.py')
    
