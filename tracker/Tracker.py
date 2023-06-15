#! /usr/bin/env python3

"""Tracker.py
class definition for tracker of turtles
"""

# TODO 
# make sure track_turtles.py works programmatically
# init
# load_model
# accept input from Detector (detections)
# accept input from path of text files (detections)
# run function
# main


from ultralytics import YOLO
import os
import glob
import cv2 as cv
import code
import numpy as np
from tracker.ImageWithDetection import ImageWithDetection
from tracker.ImageTrack import ImageTrack
from tracker.DetectionWithID import DetectionWithID
from tracker.ImageWithDetectionTrack import ImageWithDetectionTrack

from classifier.Classifier import Classifier

# load model
# model = YOLO('weights/20230430_yolov8x_turtlesonly_best.pt')

class Tracker():
    DEFAULT_classifier_weight_file = '/home/dorian/Code/turtles/turtle_tracker/classifier/weights/yolov5_classifier_exp26.pt'
    DEFAULT_yolo_dir ='/home/dorian/Code/turtles/yolov5_turtles'
    
    def __init__(self, 
                 video_file: str, 
                 save_dir: str, 
                 classifier_weights: str = DEFAULT_classifier_weight_file,
                 yolo_dir: str = DEFAULT_yolo_dir,
                 image_width: int = None, 
                 image_height: int = None):
        self.video_file = video_file # TODO can remove vid_path from function input
        self.save_dir = save_dir
        
        self.vid_name = os.path.basename(self.video_file).rsplit('.', 1)[0]
        
        self.image_suffix = '.jpg'
        self.image_height = image_height
        self.image_width = image_width # origianl video width/height
        self.classifier_model_file = classifier_weights
        self.yolo_dir = yolo_dir
        self.TurtleClassifier = Classifier(weights_file = classifier_weights,
                                      yolo_dir = yolo_dir)
        
        
    def write_track_detections(self, txt_file, boxes):
        """
        write yolov8 box to text file
        """
        
        # open text file
        with open(txt_file, 'w') as f:
            for i, id in enumerate(boxes.id):
                
                # class x1 y1 x2 y2 (normalised) conf id
                xyxyn = np.array(boxes.xyxyn[i,:])
                f.write(f'{int(boxes.cls[i])} {xyxyn[0]:.6f} {xyxyn[1]:.6f} {xyxyn[2]:.6f} {xyxyn[3]:.6f} {boxes.conf[i]:.4f} {int(boxes.id[i])}\n')

        return True
        
    # TODO plot tracks to RGB images
    # do this by making new functions in Plotter, leveraging box plots and confidence plots, but simply adding the tracking ID
    # show the track from previous frames? This is secondary


    def read_tracks_from_file(self, txt_dir, txt_search_pattern = '*.txt'):
        """
        read tracks from a text directory and return list of files and their detections and image names
        """
        
        txt_files = sorted(glob.glob(os.path.join(txt_dir, txt_search_pattern)))
        
        image_list = []
        for txt in txt_files:
            # read textfile:
            # with open(txt, 'r') as f:
            #     f.readlines()
            data = np.loadtxt(txt, dtype=float)
            
            # create ImageWithDetection object
            image_name = txt.rsplit('.', 1)[0] + self.image_suffix
            det = ImageWithDetection(txt, image_name=image_name, detection_data=data)
            
            # TODO maybe use PIL to grab image height/width and populate ImageWithDetection properties? should be redundant though when we open up the image later on anyways
            
            image_list.append(det)
        return image_list


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
    
    
    def convert_tracks_to_images(self, tracks, image_width, image_height):
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
                                                                        image_width = image_width,
                                                                        image_height = image_height,
                                                                        track_data = track_data))
                
                else:
                    # image is not new, so we take the detection data and append it!
                    index = image_name_list.index(image_name)
                    image_detection_list[index].add_detection_track(track.detections[i], track_data)
        
        # TODO: make sure image_detection_list is sorted according to image_name?
        # should be matching due to order of appearance
        return image_detection_list
    
    
    def get_tracks_from_video(self, save_dir):
        """ get tracks from video, also write each frame to jpg """
        print('tracking test')
        # model = YOLO('yolov8l.pt')
        
        # TODO put this into the __init__
        model = YOLO('/home/dorian/Code/turtles/turtle_tracker/weights/20230430_yolov8x_turtlesonly_best.pt')
        model.fuse()
        
        # img_dir = '/home/dorian/Code/turtles/turtle_datasets/job10_mini/frames_0_200'
        # img_list = glob.glob(os.path.join(img_dir, '*.PNG'))
        
        # running the model directly on the large video file will accumulate results in RAM and potentially cause out-of-memory errors
        # result = model.track(source=vid_path, save=True, persist=True)
        
        # therefore, we have to stream the video
        cap = cv.VideoCapture(self.video_file)
        
        if not cap.isOpened():
            print(f'Error opening video file: {self.video_file}')    
        
        count = 0
        MAX_COUNT = 5 # for debug purposes
        while cap.isOpened() and count <= MAX_COUNT:
            success, frame = cap.read()
            
            if not success:
                break
            
            print(f'frame: {count}')
            # TODO double check if need to convert frame to RGB
            # frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = model.track(source = frame, stream=True, persist=True, boxes=True)
            
            # save image to file for future use? not great on memory requirements/space economy
            count_str = '{:06d}'.format(count)
            save_path = os.path.join(save_dir, self.vid_name + '_frame_' + count_str + self.image_suffix)
            cv.imwrite(save_path, frame)
            
            for r in results:
                # at this point, saves image, and txt
                boxes = r.boxes
                
                # save original image height and width for box rescaling later
                self.image_height, self.image_width = r.orig_shape
                
                # create unique text file for each frame
                txt_file = os.path.join(save_dir, self.vid_name + '_frame_' + count_str + '.txt')
                
                # write detections/tracks to unique text file
                self.write_track_detections(txt_file, boxes)
                
            count += 1

        return results


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

                pred_class, predictions = TurtleClassifier.classify_image(image_crop)
                
                if not bool(pred_class): #prediction not made / confidence too low (pred_class is empty)
                    p = 0 #mark as turtle
                else: 
                    p = (int(pred_class[0]))
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
        
        overall_class_confidence_threshold = 0.5 # all class confidences must be greater than this
        overall_class_track_threshold = 0.5 # half of the track must be painted
        for i, track in enumerate(tracks):
            if self.check_overall_class_tracks(track.classifications, overall_class_track_threshold) and \
                self.check_overall_class_confidences(track.classification_confidences, overall_class_confidence_threshold):
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


    def main(self):
        
        save_txt_dir = os.path.join(self.save_dir, self.vid_name)
        os.makedirs(save_txt_dir, exist_ok=True)
        
        # get tracks into file
        self.get_tracks_from_video(save_txt_dir)
        
        # read_tracks_from_file
        image_list = self.read_tracks_from_file(txt_dir=save_txt_dir)

        
        # convert image list to tracks
        tracks = self.convert_images_to_tracks(image_list)
        
        # tracks[0].print_track()
        # TODO print/display tracks into video
        # TODO classify tracks
        self.classify_tracks(tracks)
        
        
        
        # TODO make print_track_classifications functioN;
        print('All Track Classifications:')
        for i, track in enumerate(tracks):
            print(f'track {i}: {track.classifications}')

        code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    vid_path = '/home/dorian/Code/turtles/turtle_datasets/041219-0569AMsouth/041219-0569AMsouth_trim.mp4'
    save_dir = 'output3'  
    track = Tracker(vid_path, save_dir)  
    track.main()

    
# import code
