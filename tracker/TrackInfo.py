#!/usr/bin/env python3
import copy
import numpy
from typing import List

overall_class_confidence_threshold: float = 0.5
track_history_len: int = 10

class Rect():

    def __init__(self, left: float, top: float, right: float, bottom: float) -> None:
        self.left: float = left
        self.top: float = top
        self.right: float = right
        self.bottom: float = bottom

    def get_centre(self):
        return [((self.right-self.left)*0.5)+self.left, ((self.bottom-self.top)*0.5)+self.top]


class TrackInfo():

    def __init__(self, track_id: int, latest_box: Rect, confidence: float) -> None:

        self.id: int = track_id
        self.latest_box: Rect = latest_box

        self.confidences_is_turtle: List[float] = [confidence]
        self.confidence_is_turtle_mean: float = confidence
        self.confidence_is_turtle_std_dev: float = 0.0

        self.confidences_painted: List[float] = []
        self.confidence_painted_mean: float = 0.0
        self.confidence_painted_std_dev: float = 0.0

        self.track_history: List[tuple] = [latest_box.get_centre()]

    def update_turtleness(self, latest_box: Rect, confidence: float) -> None:
        self.confidences_is_turtle.append(confidence)
        self.confidence_is_turtle_mean = numpy.mean(self.confidences_is_turtle)
        self.confidence_is_turtle_std_dev = numpy.std(self.confidences_is_turtle)

        self.latest_box = copy.copy(latest_box)
        
        if len(self.track_history) > track_history_len:
            self.track_history.pop(0)
        self.track_history.append(latest_box.get_centre())


    def update_paintedness(self, confidence: float) -> None:
        self.confidences_painted.append(confidence)
        self.confidence_painted_mean = numpy.mean(self.confidences_painted)
        self.confidence_painted_std_dev = numpy.std(self.confidences_painted)

    def latest_is_painted(self) -> bool:
        return self.confidences_painted[-1] > overall_class_confidence_threshold
  
    
    def is_painted(self) -> bool:
        return self.confidence_painted_mean > overall_class_confidence_threshold

