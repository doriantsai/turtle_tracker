#! /usr/bin/env python3

"""
class for image with detection and tracks
"""

from tracker.ImageWithDetection import ImageWithDetection
import numpy as np

class ImageWithDetectionTrack(ImageWithDetection):
    
    def __init__(self,
                 txt_file: str,
                 image_name: str,
                 detection_data,
                 image_width: int = None,
                 image_height: int = None,
                 track_data = None):
        
        # only setup to take in one detection/track at a time
        
        ImageWithDetection.__init__(self, txt_file, image_name, detection_data, image_width, image_height)
        
        self.classifications = []
        self.classification_confidences = []
        self.classification_overall = []
        self.track_data = track_data
        # TODO should only be one detection incoming at a time
        # TODO ensure that track_data and detection_data are same length, so that they correspond
        # (same number of rows)
        self.append_track(track_data)
        
        # track_id
        # overall classification
        # classification confidence
        # initial classification
        
    def append_track(self, track_data):
        self.classifications.append(track_data[0])
        self.classification_confidences.append(track_data[1])
        self.classification_overall.append(track_data[2])
        
    def add_detection_track(self, detection, track):
        self.append_detection_from_obj(detection)
        self.append_track(track)
        
    def get_detection_track_as_array(self, index, OVERALL=True):
        if OVERALL:
            det = [self.classes[index],
                self.boxes[index][0],
                self.boxes[index][1],
                self.boxes[index][2],
                self.boxes[index][3],
                self.detection_confidences[index], # detection confidence
                self.ids[index],
                self.classification_overall[index], # maybe make boolean for original classification vs overall classification
                self.classification_confidences[index]]
        else:
            det = [self.classes[index],
                self.boxes[index][0],
                self.boxes[index][1],
                self.boxes[index][2],
                self.boxes[index][3],
                self.detection_confidences[index], # detection confidence
                self.ids[index],
                self.classifications[index], # maybe make boolean for original classification vs overall classification
                self.classification_confidences[index]]
        return np.array(det)
        