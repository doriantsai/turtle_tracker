#!/usr/bin/env python3

import copy
import numpy
from typing import List

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
    def __init__(self, track_id: int, time: float, latest_box: Rect, confidence: float) -> None:
        self.id: int = track_id
        self.first_seen: float = time
        self.latest_box: Rect = latest_box

        self.confidences_is_turtle: List[float] = [confidence]
        self.confidence_is_turtle_mean: float = confidence
        self.confidence_is_turtle_std_dev: float = 0.0

        self.confidences_is_painted: List[float] = []
        self.confidence_is_painted_mean: float = 0.0
        self.confidence_is_painted_std_dev: float = 0.0

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
        self.confidences_is_painted.append(confidence)
        self.confidence_is_painted_mean = numpy.mean(self.confidences_is_painted)
        self.confidence_is_painted_std_dev = numpy.std(self.confidences_is_painted)

    def latest_is_painted(self, threshold: float) -> bool:
        return self.confidences_is_painted[-1] > threshold
  
    
    def is_painted(self, threshold: float) -> bool:
        return self.confidence_is_painted_mean > threshold

