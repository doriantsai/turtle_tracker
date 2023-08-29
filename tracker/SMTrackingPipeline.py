#!/usr/bin/env python3

import os
import sys
import csv
import yaml
from typing import List, Dict, Tuple, Optional

import numpy
from tqdm import tqdm

import cv2

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
    def __init__(self, path_config_pipeline: str, path_config_tracker: Optional[str] = None, keep_clean_view: bool = False) -> None:
        with open(os.path.expanduser(path_config_pipeline), 'r') as file:
            configuration: yaml = yaml.safe_load(file)

        if configuration is None:
            raise Exception("Unable to load pipeline configuration!")
        
        if not path_config_tracker:
            path_config_tracker = "botsorttracker_config.yaml"
        
        self.path_config_tracker = os.path.expanduser(path_config_tracker)
        if not os.path.exists(self.path_config_tracker):
            raise Exception("Unable to load tracker configuration!")
        
        try:
            self.all_detection_models: dict[str, str] = configuration['detection_models']
            self.all_classification_models: dict[str, str] = configuration['classification_models']
        except KeyError:
            raise Exception("Configuration incomplete: please specify both detection_models and classification_models!")
        
        if len(self.all_detection_models) == 0:
            raise Exception("No detection_models found in configuration!")
        
        if len(self.all_classification_models) == 0:
            raise Exception("No classification_models found in configuration!")

        self.write_video: bool = load_config_value(configuration, "write_video", True)
        self.frame_skip: int = load_config_value(configuration, "frame_skip", 2)
        self.fps: float = 30.0
        self.keep_clean_view: bool = keep_clean_view
        self.detection_confidence_threshold: float = 0.2
        self.detection_iou_threshold: float = 0.5
        self.detector_image_size: int = 640
        self.output_image_height: int = 720
        self.classifier_image_size: int = 64
        self.total_frames: int = 0
        self.frame_index: int = 0
        self.video_in: Optional[cv2.VideoCapture] = None

        self.detection_model_name: str = ""
        self.classification_model_name: str = ""

        self.tracks: dict[int, TrackInfo] = {}
        self.tracks_updated: List[TrackInfo] = []
        self.plotter: Plotter = Plotter()

    def setup(self, video_in_path: str, output_dir_path: str, detection_model_name: Optional[str] = None, classification_model_name: Optional[str] = None) -> None:
        self.frame_index = 0
        self.video_path = os.path.expanduser(video_in_path)
        self.output_dir_path = os.path.expanduser(output_dir_path)

        self.detection_model_name = detection_model_name

        if not detection_model_name:
            # Use the first model specified.
            detection_model_path = next(iter(self.all_detection_models.values()))
        else:
            detection_model_path = self.all_detection_models[detection_model_name]

        self.classification_model_name = classification_model_name

        if not classification_model_name:
            # Use the first model specified.
            classification_model_path = next(iter(self.all_classification_models.values()))
        else:
            classification_model_path = self.all_classification_models[classification_model_name]

        self.video_name: str = os.path.basename(self.video_path).rsplit('.', 1)[0]
        self.output_tracks: str = os.path.join(self.output_dir_path, self.video_name + '_tracks.csv')

        os.makedirs(self.output_dir_path, exist_ok=True)

        self.model_track: YOLO = YOLO(os.path.expanduser(detection_model_path))
        self.model_track.fuse()
        self.TurtleClassifier: Classifier = Classifier(weights_file = classification_model_path,
                                           classifier_image_size=self.classifier_image_size)

        self.setup_video_read()
        
        if self.write_video:
            self.init_video_write()

    def setup_video_read(self) -> None:        
        print(f'Video name: {self.video_name}')
        print(f'Video location: {self.video_path}')

        if self.video_in:
            self.video_in.release()

        self.video_in = cv2.VideoCapture(self.video_path)

        if not self.video_in.isOpened():
            print(f'Error opening video file: {self.video_path}')
            exit()
            
        # get fps of video
        self.fps = int(self.video_in.get(cv2.CAP_PROP_FPS))
        print(f'Video FPS: {self.fps}')
        
        # get total number of frames of video
        self.total_frames = int(self.video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Video frame count: {self.total_frames}')
        
        self.image_width = int(self.video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(self.video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        print(f'Image width: {self.image_width}')
        print(f'Image height: {self.image_height}')

        image_ratio: float = float(self.image_width) / self.image_height
        self.dimensions_view: tuple[int, int] = (int(self.output_image_height * image_ratio), self.output_image_height)

        image_scale_factor: float = self.detector_image_size / self.image_width
        self.dimensions_processing: tuple[int, int] = (int(self.image_width * image_scale_factor), int(self.image_height * image_scale_factor))

        self.mat_original: numpy.ndarray = numpy.zeros([self.image_height, self.image_width, 3], dtype=numpy.uint8)
        self.mat_turtle_finding: numpy.ndarray = numpy.zeros([self.dimensions_processing[1], self.dimensions_processing[0], 3], dtype=numpy.uint8)
        self.mat_view_processed: numpy.ndarray = numpy.zeros([self.dimensions_view[1], self.dimensions_view[0], 3], dtype=numpy.uint8)
        self.mat_view_clean: Optional[numpy.ndarray] = numpy.zeros([self.dimensions_view[1], self.dimensions_view[0], 3], dtype=numpy.uint8) if self.keep_clean_view else None

    def find_tracks_in_frame(self, time: float, frame: numpy.ndarray) -> None:
        '''Given an image as a numpy array, find and track all turtles.
        '''
        self.tracks_updated.clear()
        results: List[Results] = self.model_track.track(source=frame,
                                         stream=True, 
                                         persist=True, 
                                         boxes=True,
                                         verbose=False,
                                         conf=self.detection_confidence_threshold, # test for detection thresholds
                                         iou=self.detection_iou_threshold,
                                         tracker=self.path_config_tracker)
        
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes: Boxes = result.boxes

            for i, id in enumerate(boxes.id):
                track_id: int = int(id) # track_id starts at one :'(
                xyxyn: numpy.ndarray = numpy.array(boxes.xyxyn[i])
                latest_box: Rect = Rect(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3])
                confidence: float = float(boxes.conf[i])

                if track_id not in self.tracks.keys():
                    # Create a new track
                    new_track: TrackInfo = TrackInfo(track_id, time, latest_box, confidence)
                    self.tracks[track_id] = new_track
                    self.tracks_updated.append(new_track)
                else:
                    # Update existing track information
                    existing_track: TrackInfo = self.tracks[track_id]
                    existing_track.update_turtleness(latest_box, confidence)
                    self.tracks_updated.append(existing_track)

    def classify_turtles(self, frame: numpy.ndarray) -> None:
        (height, width, _) = frame.shape
        for track in self.tracks_updated:
            box: Rect = track.latest_box
            roi_left: int = int(box.left * width)
            roi_right: int = int(box.right * width)
            roi_top: int = int(box.top * height)
            roi_bottom: int = int(box.bottom * height)
            frame_cropped: numpy.ndarray = frame[roi_top:roi_bottom, roi_left:roi_right, :]
            cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB, frame_cropped)
            paintedness_confidence = self.TurtleClassifier.classify(frame_cropped)
            cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2BGR, frame_cropped)
            track.update_paintedness(paintedness_confidence)

    def plot_data(self, frame: numpy.ndarray) -> None:
        # plotting onto the image with self.plotter
        for track in self.tracks_updated:
            self.plotter.draw_labeled_box(frame, track)

    def init_video_write(self) -> None:
        video_out_name = os.path.join(self.output_dir_path, self.video_name + '_tracked.mp4')
        self.video_out = cv2.VideoWriter(video_out_name, 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 
                                   int(numpy.ceil(self.fps / self.frame_skip)), 
                                   self.dimensions_view,
                                   isColor=True)
        
    def write_to_csv(self) -> None:
        header = ['track_id', 'turtle_confidences', 'marked_confidences', 'marked_confidence_mean']
        with open(self.output_tracks, mode='w', newline='') as csv_file:
            f = csv.writer(csv_file)
            f.writerow(header)
            for track in self.tracks.values():
                track_id = track.id
                turtleness = track.confidences_is_turtle
                paintedness = track.confidences_is_painted
                paintedness_avg = track.confidence_is_painted_mean

                f.writerow([track_id, turtleness, paintedness, paintedness_avg])

    def process_frame(self) -> bool:
        if not self.video_in.isOpened():
            return False
        
        self.video_in.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        read_result: Tuple[bool, numpy.ndarray] = self.video_in.read(self.mat_original)

        if not read_result[0]:
            return False

        cv2.resize(src=self.mat_original, dsize=self.dimensions_processing, dst=self.mat_turtle_finding)

        cv2.resize(src=self.mat_original, dsize=self.dimensions_view, dst=self.mat_view_processed)

        if self.keep_clean_view:
            numpy.copyto(src=self.mat_view_processed, dst=self.mat_view_clean)

        time: float = self.frame_index / self.fps
        self.find_tracks_in_frame(time, self.mat_turtle_finding)
        self.classify_turtles(self.mat_original)
        self.plot_data(self.mat_view_processed)
        
        if self.write_video:
            self.video_out.write(self.mat_view_processed)

        self.frame_index += self.frame_skip
        return True
    
    def finish(self) -> None:
        cv2.destroyAllWindows()
        
        if self.write_video:
            self.video_out.release()

        self.video_in.release()

        self.write_to_csv()

    def run(self, show_preview_window: bool) -> None:
        progress_bar: tqdm = tqdm(total=self.total_frames)

        while self.process_frame():        
            if show_preview_window:
                cv2.imshow('images', self.mat_view_processed)
                if cv2.waitKey(1)& 0xFF == ord('q'):
                    print("Shutting down and saving data...")
                    break 
            
            progress_bar.update(self.frame_skip)
        
        self.finish()

    def detection_model_exists(self, detection_model_name: str) -> bool:
        return detection_model_name in self.all_detection_models.keys()
    
    def classifier_model_exists(self, classification_model_name: str) -> bool:
        return classification_model_name in self.all_classification_models.keys()


def get_kwargs(args: List[str]) -> Dict[str, str]:
    kwargs: Dict[str, str] = dict()

    for arg in args:
        split = arg.split(":=")
        if len(split) != 2:
            continue

        kwargs[split[0]] = split[1]
    
    return kwargs

def parse_bool(value: str) -> bool:
    return value.lower() in ["true", "yes", "y"]

def main() -> None:
    kwargs: Dict[str, str] = get_kwargs(sys.argv[1:])

    # Process required arguments
    try:
        video_in_path: str = kwargs["video_in_path"]
        output_path: str = kwargs["output_path"]
    except KeyError:
        print(f"Usage: {os.path.basename(__file__)} video_in_path:=/path/to/video/file output_path:=/path/to/output/directory/ [config:=/path/to/configuration.yaml]")
        return

    try:
        configuration_path: str = kwargs["config"]
    except KeyError:
        configuration_path: str = "sm_tracking_pipeline_config.yaml"

    try:
        show_preview_window: bool = parse_bool(kwargs["show_preview_window"])
    except KeyError:
        show_preview_window: bool = True

    pipeline: Pipeline = Pipeline(configuration_path)

    pipeline.setup(video_in_path, output_path)
    pipeline.run(show_preview_window)


if __name__ == "__main__":
    main()