#! /usr/bin/env python3

"""
class for images that have detections [class, x1 y1 x2 y2 conf id], where boxes xyxy are normalised to image size
and name properties
"""
import os
import glob
import numpy as np

class ImageWithDetection():
    
    def __init__(self, txt_file: str, image_name: str, detections, image_width: int = None, image_height: int = None):
        
        self.txt_file = txt_file
        self.image_name = image_name
        self.detections = np.array(detections) # [class x1 y1 x2 y2 conf id] for each row of detections, numpy array
        self.image_width = image_width
        self.image_height = image_height
        # self.class_label = detection[0]
        # self.xyxyn = detection[1:4]
        # self.conf = detection[5]
        # self.id = detection[6]
        
        self.ids = self.get_ids()
        self.classes = self.get_classes()
        self.boxes = self.get_boxes()
        self.confidences = self.get_confidences()
    
    def get_classes(self):
        self.classes = self.detections[:,0]
        
    def get_boxes(self):
        self.boxes = self.detections[:,1:5]
        
    def get_confidences(self):
        self.confidences = self.detections[:,5]
    
    def get_ids(self):
        self.ids = self.detections[:, 6]
    