#!/usr/bin/env python3

import time
import os
import sys
import csv
import yaml
from typing import List, Dict, Tuple, Optional

import numpy
from tqdm import tqdm

import cv2
from PIL import Image as PILImage

from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

from classifier.Classifierv8 import Classifier
from plotter.Plotter import Plotter

from tracker.TrackInfo import TrackInfo, Rect

def load_config_value(configuration: yaml, config_key: str, default_value: any) -> any:
    try:
        return configuration[config_key]
    except KeyError:
        return default_value


class Pipeline():
    def __init__(self, configuration_path: str) -> None:
        with open(os.path.expanduser(configuration_path), 'r') as file:
            configuration: yaml = yaml.safe_load(file)

        if configuration is None:
            raise Exception("Unable to load configuration!")
        
        try:
            self.all_detection_models: dict[str, str] = configuration['detection_models']
            self.all_classification_models: dict[str, str] = configuration['classification_models']
        except KeyError:
            raise Exception("Configuration incomplete: please specify both detection_models and classification_models!")
        
        if len(self.all_detection_models) == 0:
            raise Exception("No detection_models found in configuration!")
        
        if len(self.all_classification_models) == 0:
            raise Exception("No classification_models found in configuration!")

        self.show_preview_window: bool = load_config_value(configuration, "show_preview_window", False)
        self.write_video: bool = load_config_value(configuration, "show_preview_window", True)
        self.frame_skip: int = load_config_value(configuration, "frame_skip", 2)
        self.detection_confidence_threshold: float = 0.2
        self.detection_iou_threshold: float = 0.5
        self.detector_image_size: int = 640
        self.output_image_size = (1280, 720)
        self.classifier_image_size: int = 64

        self.paused: bool = False

        self.tracks: dict[int, TrackInfo] = {}

        self.plotter: Plotter = Plotter()

    def setup(self, video_in_path: str, output_dir_path: str, detection_model_name: Optional[str] = None, classification_model_name: Optional[str] = None) -> None:
        self.video_path = os.path.expanduser(video_in_path)
        self.output_dir_path = os.path.expanduser(output_dir_path)

        if detection_model_name is None:
            # Use the first model specified.
            detection_model_path = next(iter(self.all_detection_models.values()))
        else:
            detection_model_path = self.all_detection_models[detection_model_name]

        if classification_model_name is None:
            # Use the first model specified.
            classification_model_path = next(iter(self.all_classification_models.values()))
        else:
            classification_model_path = self.all_classification_models[classification_model_name]

        self.video_name: str = os.path.basename(self.video_path).rsplit('.', 1)[0]
        self.output_tracks: str = os.path.join(self.output_dir_path,'tracks.csv')

        os.makedirs(self.output_dir_path, exist_ok=True)

        self.model_track: YOLO = YOLO(os.path.expanduser(detection_model_path))
        self.model_track.fuse()
        self.TurtleClassifier: Classifier = Classifier(weights_file = classification_model_path,
                                           classifier_image_size=self.classifier_image_size)

        self.get_video_info()

        self.image_scale_factor: float = self.detector_image_size / self.image_width
   
    def set_playing(self, playing: bool) -> None:
        self.paused = not playing

    def get_video_info(self) -> None:        
        print(f'Video name: {self.video_name}')
        print(f'Video location: {self.video_path}')
        self.cap: cv2.VideoCapture = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f'Error opening video file: {self.video_path}')
            exit()
            
        # get fps of video
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f'Video FPS: {self.fps}')
        
        # get total number of frames of video
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Video frame count: {self.total_frames}')
        
        self.image_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        print(f'Image width: {self.image_width}')
        print(f'Image height: {self.image_height}')

    def find_tracks_in_frame(self, frame) -> List[int]:
        '''Given an image as a numpy array, find and track all turtles.
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

                if track_id not in self.tracks.keys():
                    # Create a new track
                    self.tracks[track_id] = TrackInfo(track_id, latest_box, confidence)
                else:
                    # Update existing track information
                    self.tracks[track_id].update_turtleness(latest_box, confidence)

        return track_ids_in_frame

    def classify_turtles(self, frame: numpy.ndarray, track_ids_in_frame: List[int]) -> None:
        # print(track_ids_in_frame)
        for id in track_ids_in_frame:
            # print(id, len(self.tracks))
            curr_track: TrackInfo = self.tracks[id]
            box = curr_track.latest_box
            frame_rgb = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_crop = self.TurtleClassifier.crop_image(frame_rgb,box)
            paintedness_confidence = self.TurtleClassifier.classify(image_crop)
            # print(paintedness_confidence)
            curr_track.update_paintedness(paintedness_confidence)

    def plot_data(self, frame: numpy.ndarray, track_ids_in_frame: List[int]) -> None:
        # plotting onto the image with self.plotter
        for track_id in track_ids_in_frame:
            self.plotter.draw_labeled_box(frame, self.tracks[track_id])

    def init_video_write(self) -> None:
        video_out_name = os.path.join(self.output_dir_path, self.video_name + '_tracked.mp4')
        self.video_out = cv2.VideoWriter(video_out_name, 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 
                                   int(numpy.ceil(self.fps / self.frame_skip)), 
                                   self.output_image_size,
                                   isColor=True)# writing to video
        
    def update_video(self, frame: numpy.ndarray) -> None:
        self.video_out.write(frame)
        
    def write_to_csv(self, all_tracks: Dict[int, TrackInfo]) -> None:
        header = ['track_id', 'turtleness', 'paintedness', 'paintedness_avg']
        with open(self.output_tracks, mode='w', newline='') as csv_file:
            f = csv.writer(csv_file)
            f.writerow(header)
            for i in all_tracks.keys():
                track_id = all_tracks[i].id
                turtleness = all_tracks[i].confidences_is_turtle
                paintedness = all_tracks[i].confidences_painted
                paintedness_avg = all_tracks[i].confidence_painted_mean

                write_list = [track_id, turtleness, paintedness, paintedness_avg]
                f.writerow(write_list)

    def run(self) -> None:
        frame_index: int = 0
        progress_bar: tqdm = tqdm(total=self.total_frames)
        
        if self.write_video:
            self.init_video_write()

        while self.cap.isOpened():
            if self.paused:
                time.sleep(0.5)
                continue

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            read_result: Tuple[bool, numpy.ndarray] = self.cap.read()
            read_success: bool = read_result[0]

            if not read_success:
                break

            frame: numpy.ndarray = read_result[1]

            (frame_width, frame_height, _) = frame.shape
            frame_ratio: float = float(frame_width) / frame_height
            view_height: int = self.output_image_size[1]

            # TODO: This could be made faster by allocating MatLikes of the correct sizes during initialisation and then providing them to resize.
            frame_resized: numpy.ndarray = cv2.resize(frame, None, fx=self.image_scale_factor, fy=self.image_scale_factor)
            frame_view: numpy.ndarray = cv2.resize(frame, (view_height, int(view_height * frame_ratio)))

            # TODO: We could probably return a list of the TrackInfo we have already looked up. These should be references to the original objects.
            track_ids_in_frame: List[int] = self.find_tracks_in_frame(frame_resized)
            self.classify_turtles(frame_view, track_ids_in_frame)
            self.plot_data(frame_view, track_ids_in_frame)
            
            if self.write_video:
                self.update_video(frame_view)
            
            if self.show_preview_window:
                cv2.imshow('images', frame_view)
                if cv2.waitKey(1)& 0xFF == ord('q'):
                    break 

            frame_index += self.frame_skip
            progress_bar.update(self.frame_skip)

        if self.show_preview_window:
            cv2.destroyAllWindows()

        self.write_to_csv(self.tracks)
        # Video has been processed. Release it.
        self.cap.release()

def get_kwargs(args: List[str]) -> Dict[str, str]:
    kwargs: Dict[str, str] = dict()

    for arg in args:
        split = arg.split(":=")
        if len(split) != 2:
            continue

        kwargs[split[0]] = split[1]
    
    return kwargs

def main() -> None:
    kwargs: Dict[str, str] = get_kwargs(sys.argv[1:])

    # Process required arguments
    try:
        video_in_path: str = kwargs["video_in_path"]
        output_path: str = kwargs["output_path"]
    except KeyError:
        print("Usage: SMTrackingPipeline.py video_in_path:=/path/to/video/file output_path:=/path/to/output/directory/ [config:=/path/to/configuration.yaml]")
        return

    try:
        configuration_path: str = kwargs["config"]
    except KeyError:
        configuration_path: str = "sm_tracking_pipeline_config.yaml"    

    pipeline: Pipeline = Pipeline(configuration_path)

    pipeline.setup(video_in_path, output_path)
    pipeline.run()


if __name__ == "__main__":
    main()