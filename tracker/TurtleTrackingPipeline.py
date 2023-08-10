import cv2 as cv
import os
import glob
import code
import numpy as np
from ultralytics import YOLO
import time
import yaml
from PIL import Image as PILImage
import csv
from datetime import datetime

from tracker.ImageTrack import ImageTrack
from tracker.DetectionWithID import DetectionWithID
from tracker.ImageWithDetectionTrack import ImageWithDetectionTrack

from classifier.Classifier import Classifier

from plotter.Plotter import Plotter
from tracker.ImageWithDetection import ImageWithDetection
# from tracker.Tracker import Tracker

'''Class that takes a video and outputs a video file with tracks and
classifications and a text file with final turtle counts'''



class Pipeline:

    default_config_file = 'pipeline_config.yaml' # configuration file for video/model/output
    default_output_file = 'turtle_counts.csv'
    default_image_suffix = '.jpg'
    img_scale_factor = 0.3 # for display-purposes only
    max_time_min = 6 # max 6 minutes/video
    default_max_frames = 1000
    
    def __init__(self,
                 config_file: str = default_config_file,
                 img_suffix: str = default_image_suffix,
                 output_file: str = default_output_file,
                 max_frames = None):
        
        self.config_file = config_file
        config = self.read_config(config_file)
        
        self.video_path = config['video_path_in']
        self.video_name = os.path.basename(self.video_path).rsplit('.', 1)[0]
        
        self.save_dir = config['save_dir']
        self.save_frame_dir = os.path.join(self.save_dir, 'frames')
        
        # make output file default to video name, not default_output_file
        output_name = self.video_name + '.csv'
        self.output_file = os.path.join(self.save_dir, output_name)
        
        
        self.frame_skip = config['frame_skip']
        # self.fps = 29 # default fps expected from the drone footage
        
        self.get_video_info()
        
        if max_frames is None or max_frames <= 0:
            self.set_max_count(self.max_time_min) # setup max count from defaults
        else:
            self.max_count = max_frames
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_frame_dir, exist_ok=True)
        
        self.image_suffix = img_suffix
        
        self.model_track = YOLO(config['detection_model_path'])
        self.model_track.fuse()
        self.classifier_weights = config['classification_model_path']
        self.yolo_path = config['YOLOv5_install_path']
        
        self.overall_class_confidence_threshold = config['overall_class_confidence_threshold']
        self.overall_class_track_threshold = config['overall_class_track_threshold']

        self.detection_confidence_threshold = config['detection_confidence_threshold']
        self.detection_iou_threshold = config['detection_iou_threshold']

        self.TurtleClassifier = Classifier(weights_file = self.classifier_weights,
                                      yolo_dir = self.yolo_path)


    def get_max_count(self):
        return self.max_count
    
    
    def set_max_count(self, max_time_min = 6):
        # arbitrarily large number for very long videos (5 minutes, fps)
        self.max_count = int(max_time_min * 60 * self.fps)
    
    
    def make_video(self,
                  name_vid_out : str,
                  transformed_imglist: list,
                  video_in_location: str):
        '''Given new video name, a list of transformed frames and a video based
        off, outputs a video'''
        vidcap = cv.VideoCapture(video_in_location)
        w = int(vidcap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(vidcap.get(cv.CAP_PROP_FRAME_HEIGHT))
        video_out = video_in_location.split('.')[0] + name_vid_out + '.mp4'
        out = cv.VideoWriter(video_out, 
                              cv.VideoWriter_fourcc(*"mp4v"), 
                              self.fps, 
                              (w, h), 
                              isColor=True)
        for image in transformed_imglist:
            out.write(image)
        out.release()


    def save_turtle_total_count(self, 
                             txt_file: str, 
                             T_count, 
                             P_count):
        '''Given turtle and painted turtle count, write txt file with turtle and
        painted turtle count'''
        with open(txt_file, 'w') as f:
            f.write(f'{T_count} {P_count}')
        return T_count, P_count


    def get_tracks_from_frame(self, frame):
        '''Given an image in a numpy array, find and track a turtle.
        Returns an array of numbers with class,x1,y1,x2,y2,conf,track_id with
        x1,y1,x2,y2 all resized for the image'''
        # [cls, x1 y1 x2 y2 conf, track_id, predicted class, classification_confidence]
        no_detection_case = [np.array([0, 0, 0.1, 0, 0.1, 0, -1, 0, 0.0])]
        box_array = []
        results = self.model_track.track(source=frame, 
                                         stream=True, 
                                         persist=True, 
                                         boxes=True,
                                         verbose=False,
                                         conf=self.detection_confidence_threshold, # test for detection thresholds
                                         iou=self.detection_iou_threshold,
                                         tracker='botsorttracker_config.yaml')
        # code.interact(local=dict(globals(), **locals()))
        # if len(results) == 0:
        #     return no_detection_case
        # if len(results) > 0:
        for r in results:
            boxes = r.boxes
            
            # no detection case
            if boxes.id is None:
                return box_array.append(no_detection_case) # assign turtle with -1 id for no detections

            for i, id in enumerate(boxes.id):
                xyxyn = np.array(boxes.xyxyn[i, :])
                box_array.append([int(boxes.cls[i]),            # class
                                float(xyxyn[0]),    # x1
                                float(xyxyn[1]),    # y1
                                float(xyxyn[2]),    # x2
                                float(xyxyn[3]),    # y2
                                float(boxes.conf[i]),         # conf
                                int(boxes.id[i])])            # track id
        return box_array
        

    def get_video_info(self):
        """ get video info """
        
        print(f'video name: {self.video_name}')
        print(f'video location: {self.video_path}')
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f'Error opening video file: {self.video_path}')
            exit()
            
        # get fps of video
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        print(f'Video FPS: {self.fps}')
        
        # get total number of frames of video
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print(f'Video frame count: {total_frames}')
        
        # open cap and read just one image
        count = 0
        while cap.isOpened() and count <= 1:
            success, frame = cap.read()
            if not success:
                cap.release() # release object
                break
            imgw, imgh = frame.shape[1], frame.shape[0]
        
            self.image_width = imgw
            self.image_height = imgh
            count += 1
        
        cap.release()
        print(f'image width: {self.image_width}')
        print(f'image height: {self.image_height}')
        

    def get_tracks_from_video(self, SHOW=False): 
        ''' Given video file, get tracks across entire video
        Returns list of image tracks (ImageTrack object)
        MAX_COUNT = maximum number of frames before video closes
        '''
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f'Error opening video file: {self.video_path}')
            exit()

        
        print(f'Frame skip interval: {self.frame_skip}')
        
        start_read_time = time.time()
        
        count = 0
        image_detection_list = []
        
        while cap.isOpened() and count <= self.max_count:
            success, frame = cap.read()
            if not success:
                cap.release() # release object
                break
        
            # skip frames based on FRAME_SKIP
            if count % self.frame_skip == 0:
                print(f'frame: {count}')
                
                # sadly, write frame to file, as we need them indexed for the classification during tracks
                # TODO try to find a way without so much read/write to disk?
                count_str = '{:06d}'.format(count)
                image_name = self.video_name + '_frame_' + count_str + self.image_suffix
                
                # TODO remove this with in-frame detection, tracking and classification
                save_path = os.path.join(self.save_frame_dir, image_name)
                # cv.imwrite(save_path, frame)
                
                # track and detect single frame
                # [class,x1,y1,x2,y2,conf,track_id] with x1,y1,x2,y2 all resized for the image
                box_list = self.get_tracks_from_frame(frame)

                # if count == 6944: # TODO unsure why in no detections, box_list becomes Nonetype
                #     print(f'arrived at count - to stop, boxlist is Nonetype?')
                #     code.interact(local=dict(globals(), **locals()))
                    
                # for each detection, run classifier
                box_array_with_classification = []
                if type(box_list) == type(None):
                    no_detection_case = [np.array([0, 0, 0.1, 0, 0.1, 0, -1, 0, 0.0])]
                    box_array_with_classification = no_detection_case
                else: 
                    for box in box_list:
                        # classifer works on PIL images currently
                        # TODO change to yolov8 so no longer require PIL image - just operate on numpy arrays
                        
                        frame_rgb = PILImage.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                        # image_rgb.show()
                        
                        image_crop = self.TurtleClassifier.crop_image(frame_rgb, box[1:5], self.image_width, self.image_height)
                        
                        predicted_class, predictions = self.TurtleClassifier.classify_image(image_crop)
                        # append classifications to the det object
                        # det.append_classification(predicted_class, 1-predictions[predicted_class].item())
                        box.append(predicted_class)
                        box.append(1-predictions[predicted_class].item())
                        box_array_with_classification.append(box)
                        
                
                # det = ImageWithDetection('empty', save_path, box_array_with_classification, self.image_width, self.image_height)
                det = ImageWithDetection(txt_file='empty', 
                                        image_name=save_path,
                                        detection_data=box_array_with_classification,
                                        image_width=self.image_width,
                                        image_height=self.image_height)

                    
                # else detections, classifications are empty 
                image_detection_list.append(det)
                
            if SHOW:
                img = cv.resize(frame, None, fx=self.img_scale_factor,
                                fy=self.img_scale_factor, interpolation=cv.INTER_AREA)
                cv.imshow('images', img)
                cv.waitKey(0)   
            
            count += 1
            
        # release the video capture object
        cap.release()
        
        end_read_time = time.time()
        sec = end_read_time - start_read_time
        print('video read time: {} sec'.format(sec))
        print('video read time: {} min'.format(sec / 60.0))
        
        return image_detection_list


    def count_painted_turtles(self, tracks):
        '''count painted turtle tracks, ignoring the classified overall'''
        painted_count = 0
        unpainted_count = 0
        painted_list = []
        turtle_list = []
        for i, track in enumerate(tracks):
            for j in enumerate(track.classifications):
                if j[1] == 0 and track.id not in painted_list:
                    painted_list.append(track.id)
                    painted_count += 1
                elif j[1] == 1 and track.id not in turtle_list:
                    turtle_list.append(track.id)
                    unpainted_count += 1
                # NOTE will count a flickering turtle as both painted and unpainted
        return painted_count, unpainted_count


    def count_painted_turtles_overall(self, tracks):
        """ count painted turtle tracks, tracks must be classified overall """
        painted_count = 0
        unpainted_count = 0
        for i, track in enumerate(tracks):
            if track.classification_overall:
                painted_count += 1
            else:
                unpainted_count += 1
        return painted_count, unpainted_count
    
    
    def make_video_after_tracks(self, image_detection_track_list):
        """ make video of detections after tracks classified, etc """
        
        print('make_video_after_tracks')
        
        # read in the video frames
        vidcap = cv.VideoCapture(self.video_path)
        if not vidcap.isOpened():
            print(f'Error opening video file: {self.video_path}')
            exit()
        
        w = int(vidcap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(vidcap.get(cv.CAP_PROP_FRAME_HEIGHT))
        plotter = Plotter(w, h)
        
        # setup video writer
        video_out = os.path.join(self.save_dir, self.video_name + '_tracked' + '.mp4')
        out = cv.VideoWriter(video_out, 
                             cv.VideoWriter_fourcc(*"mp4v"), 
                             self.fps, 
                             (w, h), 
                             isColor=True)
        
        count = 0
        track_index = 0
            
        # iterate over video
        while vidcap.isOpened() and count <= self.max_count:
            success, frame = vidcap.read()
            if not success:
                vidcap.release()
                break
            
            if count % self.frame_skip == 0:
                print(f'writing frame: {count}')
                # apply detections/track info to frame
                #    [class, x1,y1,x2,y2, confidence, track id, classifier class, conf class]
                if len(image_detection_track_list) > 0:
                    
                    image_data = image_detection_track_list[track_index]
                    box_array = [image_data.get_detection_track_as_array(i, OVERALL=True) for i in range(len(image_data.detections))]
                    
                    # make plots
                    plotter.boxwithid(box_array, frame)
                
                # save to image writer
                out.write(frame)
                
                track_index += 1
                # TODO save different box arrays frame by frame (overall = false and overall = true) into different folders so can contrast

            count += 1

        out.release()
        vidcap.release()
    
    
    def convert_images_to_tracks(self, image_list):
        """ convert image detections to tracks"""
        
        # iterate through image_list and add/append onto tracks
        tracks = []
        track_ids = []
        # for each image_list, add a new track whenever we have new ID
        # when going through each detection, if old ID, then append detection info
        for image in image_list:
            for detection in image.detections:
                if detection.id not in track_ids:
                    # new id, thus add to track_ids and create new track and append it to list of tracks
                    tracks.append(ImageTrack(detection.id, detection))
                    track_ids.append(detection.id) # not sure if better to maintain a list of track_ids in parallel, or simply make list when needed
                    
                else:
                    # detection id is in track_ids, so we add to existing track
                    # first, find corresponding track index
                    track_index = track_ids.index(detection.id)
                    # then we add in the new detection to the same ID
                    tracks[track_index].add_detection(detection)
        
        return tracks
    
    
    def classify_tracks(self, tracks):
        """ classify tracks for painted/unpainted turtles """
        print('classifying tracks')
        # load classifier model
        # for each track:
            # read in image list/image names
            # transform each image
            # classify on each image
            # view classifier results/sum them, see any flickering?
        
        TurtleClassifier = self.TurtleClassifier
        for i, track in enumerate(tracks):
            print(f'track: {i+1}/{len(tracks)}')
            
            for j, image_name in enumerate(track.image_names):
                # need to crop image for classifier
                image = TurtleClassifier.read_image(image_name)
                image_crop = TurtleClassifier.crop_image(image, track.boxes[j], self.image_width, self.image_height)
                p, predictions = TurtleClassifier.classify_image(image_crop)
                # append classifications to track
                track.add_classification(p, 1-predictions[p].item())
    
        return tracks
    
    def classify_tracks_overall(self, tracks):
        """ classify trackers overall after per-image classification has been done """
        
        # each track has a classification and a classification_confidence
        
        # what defines the overall classification of painted (1) vs not painted (0)
        # arbitrarily:
        # if over 50% of the tracks are ID'd as painted, then the overall track is painted
        # else default to not painted
        
        # overall_class_confidence_threshold = 0.5 # all class confidences must be greater than this
        # overall_class_track_threshold = 0.5 # half of the track must be painted
        for i, track in enumerate(tracks):
            if self.check_overall_class_tracks(track.classifications, self.overall_class_track_threshold) and \
                self.check_overall_class_confidences(track.classification_confidences, self.overall_class_confidence_threshold):
                    track.classification_overall = 1 # painted turtle
            else:
                track.classification_overall = 0 # unpainted turtle
                
        return tracks
    
    
    def check_overall_class_tracks(self, class_track, overall_threshold=0.5):
        classes_per_image = np.array(class_track)
        if np.sum(classes_per_image) / len(classes_per_image) > overall_threshold:
            return True
        else:
            return False

    
    def check_overall_class_confidences(self, conf_track, threshold=0.5):
        conf_per_image = np.array(conf_track)
        if np.all(conf_per_image > threshold):
            return True
        else:
            return False
    
    def convert_tracks_to_images(self, tracks):
        """ convert tracks to image detections """
        
        # create empty lists of text for each image file:
        image_name_list = []
        image_detection_list = []
        
        # iterate through tracks and add/append onto images
        for track in tracks:
            # iterate through each detection in the track
            for i, image_name in enumerate(track.image_names):
                track_data = [track.classifications[i], 
                                  track.classification_confidences[i], 
                                  track.classification_overall]
                # if new image, we add to the image list, and then add in the corresponding detection
                if image_name not in image_name_list:
                
                    image_name_list.append(image_name)
                    txt_file = image_name.rsplit('.', 1)[0] + '.txt'
                    # TODO should go into a `track_labels' folder
                    image_detection_list.append(ImageWithDetectionTrack(txt_file=txt_file,
                                                                        image_name = image_name,
                                                                        detection_data=track.detections[i],
                                                                        image_width = self.image_width,
                                                                        image_height = self.image_height,
                                                                        track_data = track_data))
                
                else:
                    # image is not new, so we take the detection data and append it!
                    index = image_name_list.index(image_name)
                    image_detection_list[index].add_detection_track(track.detections[i], track_data)
        
        # TODO: make sure image_detection_list is sorted according to image_name?
        # should be matching due to order of appearance
        return image_detection_list
    
    
    def run(self, SHOW=False):

        start_time = time.time()

        # get detection list for each image
        image_detection_list = self.get_tracks_from_video(SHOW)
        # NOTE: we don't need to save each frame, because each frame is already 
        # just want to save the detections/metadata to a file for replotting
        # and we re-open the video when it's time to make the video with detections/plots
        
        # should be input, also slight misnomer perhaps because Yolov8 tracking actually happens in GetTracksFromVideo function
        # TODO merge into tracking pipeline
        # tracker_obj = Tracker(self.video_path, 
        #                       self.save_dir,
        #                       classifier_weights=self.classifier_weights, 
        #                       yolo_dir=self.yolo_path,
        #                       image_width=image_width, 
        #                       image_height=image_height,
        #                       overall_class_confidence_threshold=self.overall_class_confidence_threshold,
        #                       overall_class_track_threshold=self.overall_class_track_threshold)
        
        # code.interact(local=dict(globals(), **locals()))
        
        # convert from image list detections to tracks
        tracks = self.convert_images_to_tracks(image_detection_list)
        
        # run classifier on tracks
        # NOTE: requires the actual image!
        # tracks_classified = self.classify_tracks(tracks)
        
        # run classification overall on classified tracks
        tracks_overall = self.classify_tracks_overall(tracks)        
        
        # convert tracks back into image detections!
        image_detection_track_list = self.convert_tracks_to_images(tracks_overall)
        
        # plot classified tracks to file by re-opening the video and applying our tracks back to the images
        self.make_video_after_tracks(image_detection_track_list)
        
        # for overall counts of painted turtles:
        painted, unpainted = self.count_painted_turtles_overall(tracks_overall)
        print("Overal counts")
        print(f'painted count: {painted}')
        print(f'unpainted count: {unpainted}')
        print(f'total turtles: {len(tracks)}')
        
        # painted, unpainted = self.count_painted_turtles(tracks)
        # print("Count along the way")
        # print(f'painted count: {painted}')
        # print(f'unpainted count: {unpainted}')
        
        print('counting complete')
        end_time = time.time()
        sec = end_time - start_time
        print('compute time: {} sec'.format(sec))
        print('compute time: {} min'.format(sec / 60.0))
        print('compute time: {} hrs'.format(sec / 3600.0))
        
        print(f'Number of frames processed: {len(image_detection_list)}')
        print(f'Seconds/frame: {sec  / len(image_detection_list)}')
                
        self.write_counts_to_file(os.path.join(self.save_dir, self.output_file), painted, unpainted, len(tracks))
                                  
        
        
            
        return tracks_overall


    def read_config(self, config_file):
        """_summary_

        Args:
            config_file (_type_): _description_
        """
        
        with open(config_file, 'r') as file:
            yaml_data = yaml.safe_load(file)
            
        # Extract the variables
        config = {'video_path_in': yaml_data['video_path_in'],
                  'save_dir': yaml_data['save_dir'],
                  'detection_model_path': yaml_data['detection_model_path'],
                  'classification_model_path': yaml_data['classification_model_path'],
                  'YOLOv5_install_path': yaml_data['YOLOv5_install_path'],
                  'frame_skip': yaml_data['frame_skip'],
                  'detection_confidence_threshold': yaml_data['detection_confidence_threshold'],
                  'detection_iou_threshold': yaml_data['detection_iou_threshold'],
                  'overall_class_confidence_threshold': yaml_data['overall_class_confidence_threshold'],
                  'overall_class_track_threshold': yaml_data['overall_class_track_threshold']}
        
        return config


    def write_counts_to_file(self, output_file, count_painted, count_unpainted, count_total):
        """write_counts_to_file

        Args:
            output_file (str): absolute filepath to where we want to save the file
        """
        
        title_row = ['Raine AI Turtle Counts']
        label_vid = ['Video name']
        label_date = ['Date counted']
        datestr = datetime.now()
        date_counted = [datestr.strftime("%Y-%m-%d")]
        label_counts = ['painted', 'unpainted', 'total']
        counts = [count_painted, count_unpainted, count_total]
        
        # also output yaml file (configuration parameters to the csv)
        with open(self.config_file, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        
        with open(output_file, mode='w', newline='') as csv_file:
            f = csv.writer(csv_file)
            f.writerow(title_row)
            f.writerow([label_vid, self.video_name])
            f.writerow([label_date, date_counted])
            
            for i in range(len(counts)):
                f.writerow([label_counts[i], counts[i]])
            
            
            header = ['pipeline_config.yaml']
            f.writerow(header)
            for key, value in yaml_data.items():
                f.writerow([key, value])
            
        print(f'Counts written to {output_file}')
            
        
        
        
if __name__ == "__main__":
    
    
    config_file = 'pipeline_config.yaml' # locally-referenced from cd: tracker folder
    p = Pipeline(config_file=config_file, max_frames=0)
    # p = Pipeline(config_file=config_file)
    results = p.run()
    # txt_name = '/home/dorian/Code/turtles/turtle_datasets/tracking_output/test.txt'
    # p.SaveTurtleTotalCount(txt_name, T_count, P_count)

    # p.MakeVideo('031216amnorth', transformed_imglist)
    
    # code.interact(local=dict(globals(), **locals()))

