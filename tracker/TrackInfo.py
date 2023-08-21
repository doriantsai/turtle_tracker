#!/usr/bin/env python3
import copy
import numpy
from typing import List

overall_class_confidence_threshold: float = 0.7
class Rect():

    def __init__(self, left: float, top: float, right: float, bottom: float) -> None:
        self.left: float = left
        self.top: float = top
        self.right: float = right
        self.bottom: float = bottom

class TrackInfo():

    def __init__(self, track_id: int, latest_box: Rect, confidence: float) -> None:
        self.frames_detected: int = 1

        self.confidences_is_turtle: List[float] = [confidence]
        self.confidence_is_turtle_mean: float = confidence
        self.confidence_is_turtle_std_dev: float = 0.0

        self.confidences_painted: List[float] = []
        self.confidence_painted_mean: float = 0.0
        self.confidence_painted_std_dev: float = 0.0

        self.id: int = track_id
        self.latest_box: Rect = latest_box

    def update_turtleness(self, latest_box: Rect, confidence: float) -> None:
        self.confidences_is_turtle.append(confidence)
        self.confidence_is_turtle_mean = numpy.mean(self.confidences_is_turtle)
        self.confidence_is_turtle_std_dev = numpy.std(self.confidences_is_turtle)

        self.latest_box = copy.copy(latest_box)
        self.frames_detected += 1

    def update_paintedness(self, confidence: float) -> None:
        self.confidences_painted.append(confidence)
        self.confidence_painted_mean = numpy.mean(self.confidences_painted)
        self.confidence_painted_std_dev = numpy.std(self.confidences_painted)

    def is_painted(self) -> bool:
        return self.confidence_painted_mean > overall_class_confidence_threshold

