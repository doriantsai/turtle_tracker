#!/usr/bin/env python3

import time
from typing import Tuple
import cv2
import os
import sys
import numpy

from PIL import Image as PILImage

from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

from classifier.Classifierv8 import Classifier
from plotter.Plotter import Plotter

from tracker.TrackInfo import TrackInfo, Rect

from typing import List

# object detection model
detection_model_path: str = '/home/serena/Data/Turtles/yolov8x_models/train17/weights/best.pt'

# classification model
classification_model_path: str = '/home/serena/Data/Turtles/classifier_models/20230821_yolov8s-cls_best.pt'

# frame skip
frame_skip: int = 2

# detection confidence
detection_confidence_threshold: float = 0.35

# detection iou threshold
detection_iou_threshold: float = 0.5

# overall classification confidence
overall_class_confidence_threshold: float = 0.7

# overall class track threshold
overall_class_track_threshold: float = 0.5

# detection/tracker image size
detector_image_size: int = 640

# classification image size
classifier_image_size: int = 64

video_path_in: str = sys.argv[1]

save_dir: str = sys.argv[2]

class Pipeline():

    def __init__(self) -> None:

        self.SHOW: bool = True
        self.WRITE_VID: bool = False
        self.video_path: str = video_path_in
        self.video_name: str = os.path.basename(self.video_path).rsplit('.', 1)[0]
        self.save_dir: str = save_dir

        output_name: str = self.video_name + '.csv'
        self.output_file: str = os.path.join(self.save_dir, output_name)
        self.output_tracks: str = os.path.join(self.save_dir,'tracks.csv')

        self.frame_skip: int = frame_skip

        os.makedirs(self.save_dir, exist_ok=True)

        self.model_track: YOLO = YOLO(detection_model_path)
        self.model_track.fuse()
        self.TurtleClassifier: Classifier = Classifier(weights_file = classification_model_path,
                                           classifier_image_size=classifier_image_size)
        
        self.overall_class_confidence_threshold: float = overall_class_confidence_threshold
        self.overall_class_track_threshold: float = overall_class_track_threshold

        self.detection_confidence_threshold: float = detection_confidence_threshold
        self.detection_iou_threshold: float = detection_iou_threshold
        
        self.detector_image_size: int = detector_image_size

        self.get_video_info()

        self.image_scale_factor: float = self.detector_image_size / self.image_width
        
        self.classifier_image_size: int = classifier_image_size     

        self.tracks: List[TrackInfo] = []

        self.plotter: Plotter = Plotter(self.image_width, self.image_height)


    def get_video_info(self) -> None:        
        print(f'video name: {self.video_name}')
        print(f'video location: {self.video_path}')
        self.cap: cv2.VideoCapture = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f'Error opening video file: {self.video_path}')
            exit()
            
        # get fps of video
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f'Video FPS: {self.fps}')
        
        # get total number of frames of video
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Video frame count: {total_frames}')
        
        self.image_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        print(f'image width: {self.image_width}')
        print(f'image height: {self.image_height}')    
    

    def find_tracks_in_frame(self, frame) -> List[int]:
        '''Given an image in a numpy array, find and track all turtles.
        '''
        track_ids_in_frame: List[int] = []
        results: List[Results] = self.model_track.track(source=frame, 
                                         stream=True, 
                                         persist=True, 
                                         boxes=True,
                                         verbose=False,
                                         conf=self.detection_confidence_threshold, # test for detection thresholds
                                         iou=self.detection_iou_threshold,
                                         tracker='botsorttracker_config.yaml')
        
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes: Boxes = result.boxes

            for i, id in enumerate(boxes.id):
                track_id: int = int(id) # track_id starts at one :'(
                xyxyn: numpy.ndarray = numpy.array(boxes.xyxyn[i])
                latest_box: Rect = Rect(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3])
                confidence: float = float(boxes.conf[i])
                track_ids_in_frame.append(track_id)

                if track_id >= len(self.tracks)-1:
                    # Create a new track
                    self.tracks.append(TrackInfo(track_id, latest_box, confidence))
                else:
                    # Update existing track information
                    self.tracks[track_id - 1].update_turtleness(latest_box, confidence)

        return track_ids_in_frame


    def classify_turtles(self, frame: numpy.ndarray, track_ids_in_frame: List[int]) -> None:
        print(track_ids_in_frame)
        for id in track_ids_in_frame:
            print(id, len(self.tracks))
            curr_track: TrackInfo = self.tracks[id - 1]
            box = curr_track.latest_box
            frame_rgb = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_crop = self.TurtleClassifier.crop_image(frame_rgb, box, self.image_width, self.image_height)
            paintedness_confidence = self.TurtleClassifier.classify(image_crop)
            print(paintedness_confidence)
            curr_track.update_paintedness(paintedness_confidence)

    def plot_data(self, frame: numpy.ndarray, track_ids_in_frame: List[int]) -> None:
        # plotting onto the image with self.plotter
        for track_id in track_ids_in_frame:
            self.plotter.draw_labeled_box(frame, self.tracks[track_id - 1])
        
    def update_video(self, frame: numpy.ndarray) -> None:
        # writing to video
        pass

    def run(self) -> None:
        frame_index: int = 0

        while self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            read_result: Tuple[bool, numpy.ndarray] = self.cap.read()
            read_success: bool = read_result[0]

            if not read_success:
                break

            frame: numpy.ndarray = read_result[1]
            frame_resized: numpy.ndarray = cv2.resize(frame, None, fx=self.image_scale_factor, fy=self.image_scale_factor)

            track_ids_in_frame: List[int] = self.find_tracks_in_frame(frame_resized)
            self.classify_turtles(frame_resized, track_ids_in_frame)
            self.plot_data(frame, track_ids_in_frame)
            
            if self.WRITE_VID:
                self.update_video(frame)
            
            if self.SHOW:
                cv2.imshow('images', frame)
                if cv2.waitKey(1)& 0xFF == ord('q'):
                    break 

            frame_index += self.frame_skip
        # Video has been processed. Release it.
        self.cap.release()


def main() -> None:
    Pipeline().run()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

