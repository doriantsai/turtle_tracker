#! /usr/bin/env python3

"""Plotter.py
A collection of common plotting functions used by the Detector, Tracker and Classifier
"""

import os
import numpy
import cv2
from tracker.TrackInfo import TrackInfo

class Plotter:
    
    # BGR
    Green = (8, 204, 24)
    LightGreen = (154, 245, 177)
    Orange = (8, 103, 204)
    LightOrange = (115, 194, 255)
    White = (255,255,255)
    def __init__(self):
       
        self.font = cv2.FONT_HERSHEY_SIMPLEX   

    def draw_history(self, img: numpy.ndarray, track: TrackInfo):
        (image_height, image_width, depth) = img.shape
        points = numpy.array(track.track_history) * [image_width, image_height]
        points = points.astype(numpy.int32).reshape(-1,1,2)
        colour = self.LightOrange if track.is_painted() else self.LightGreen
        cv2.polylines(img, [points], isClosed=False, color=colour, thickness=2)


    def draw_label(self, img, x1, y1, text, colour, thickness):
        '''
        Given a img, starting x,y cords, text and colour, create a filled in box of specified colour
        and write the text
        '''
        #font_scale = 0.5 # 
        font_scale = max(0.5,0.0000005*max(img.shape))
        p = 5 #padding
        text_size, _ = cv2.getTextSize(text, self.font, font_scale, thickness)
        cv2.rectangle(img, (x1-p, y1-p), (x1+text_size[0]+p, y1-text_size[1]-(2*p)), colour, -1)
        cv2.putText(img, text, (x1,y1-10), self.font, font_scale, self.White, thickness)
        

    def draw_labeled_box(self, frame: numpy.ndarray, track: TrackInfo, line_thickness=1):
        '''
        Crete a box with specific string and colour.
        NOTE box coordinates come in as normalised!
        '''
        colour = self.Orange if track.latest_is_painted() else self.Green
        (image_height, image_width, depth) = frame.shape
        # import code
        # code.interact(local=dict(globals(), **locals()))
        # confidence_is_turtle: str = format(track.confidences_is_turtle[-1] * 100.0, '.0f')
        confidence_is_painted: str = format(track.confidences_painted[-1] * 100.0, '.0f')
        label: str = '{} C:{}'.format(track.id, confidence_is_painted)
        
        x1: int = int(track.latest_box.left * float(image_width))
        y1: int = int(track.latest_box.top * float(image_height))
        x2: int = int(track.latest_box.right * float(image_width))
        y2: int = int(track.latest_box.bottom * float(image_height))
        self.draw_history(frame, track)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, line_thickness) #box around turtle
        self.draw_label(frame, x1, y1, label, colour, line_thickness)
        


if __name__ == "__main__":
    
    print('Plotter.py')
    
