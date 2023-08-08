import cv2 as cv
import os
import glob
import code
import numpy as np
from ultralytics import YOLO
import time
import yaml

from plotter.Plotter import Plotter
from tracker.ImageWithDetection import ImageWithDetection
from tracker.Tracker import Tracker

'''Class that takes a video and outputs a video file with tracks and
classifications and a text file with final turtle counts'''

# TODO add method of skipping frames from video file (stride)
# TODO break-up video into 5-minute chunks (maybe ask Gavin to do this for me)

class Pipeline:

    default_config_file = 'pipeline_config.yaml' # configuration file for video/model/output
    default_image_suffix = '.jpg'
    img_scale_factor = 0.3 # for display-purposes only
    
    
    def __init__(self,
                 config_file: str = default_config_file,
                 img_suffix: str = default_image_suffix):
        
        # config = {'video_path_in': yaml_data['video_path_in'],
        #           'save_dir': yaml_data['save_dir'],
        #           'detection_model_path': yaml_data['detection_model_path'],
        #           'classification_model_path': yaml_data['classification_model_path'],
        #           'YOLOv5_install_path': yaml_data['YOLOv5_install_path']}
        
        self.config_file = config_file
        config = self.read_config(config_file)
        
        self.video_path = config['video_path_in']
        self.save_dir = config['save_dir']
        self.save_frame_dir = os.path.join(self.save_dir, 'frames')
        
        self.frame_skip = config['frame_skip']
        self.fps = 29 # default fps expected from the drone footage
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_frame_dir, exist_ok=True)
        
        self.video_name = os.path.basename(self.video_path).rsplit('.', 1)[0]
        self.image_suffix = img_suffix
        
        self.model_track = YOLO(config['detection_model_path'])
        self.model_track.fuse()
        self.classifier_weights = config['classification_model_path']
        self.yolo_path = config['YOLOv5_install_path']
        #self.classifier = Classifier(weights_file=self.classifier_weights, 
              #                       yolo_dir=yolo_path, 
               #                      confidence_threshold=0.8)


    def get_max_count(self):
        return self.max_count
    
    
    def set_max_count(self, max_time_min = 5):
        # arbitrarily large number for very long videos (5 minutes, fps)
        self.max_count = int(max_time_min * 60 * self.fps)
    
    
    def MakeVideo(self,
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


    def SaveTurtleTotalCount(self, 
                             txt_file: str, 
                             T_count, 
                             P_count):
        '''Given turtle and painted turtle count, write txt file with turtle and
        painted turtle count'''
        with open(txt_file, 'w') as f:
            f.write(f'{T_count} {P_count}')
        return T_count, P_count


    def GetTracksFromFrame(self, frame, imgw, imgh):
        '''Given an image in a numpy array, find and track a turtle.
        Returns an array of numbers with class,x1,y1,x2,y2,conf,track id with
        x1,y1,x2,y2 all resized for the image'''
        box_array = []
        results = self.model_track.track(
            source=frame, stream=True, persist=True, boxes=True)
        for r in results:
            boxes = r.boxes
            # TODO NoneType is not iterable, so need a "no detections" case
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


    def GetTracksFromVideo(self, SHOW=False, MAX_COUNT=0, MAX_TIME_MIN=5): 
        ''' Given video file, get tracks across entire video
        Returns list of image tracks (ImageTrack object)
        MAX_COUNT = maximum number of frames before video closes
        '''
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
        
        print(f'Frame skip interval: {self.frame_skip}')
        
        start_read_time = time.time()
        
        count = 0
        if MAX_COUNT == 0:
            self.set_max_count()    
        
        image_detection_list = []
        
        while cap.isOpened() and count <= MAX_COUNT :
            success, frame = cap.read()
            if not success:
                cap.release() # release object
                break
        
            # skip frames based on FRAME_SKIP
            if count % self.frame_skip == 0:
                print(f'frame: {count}')
                imgw, imgh = frame.shape[1], frame.shape[0]
                
                # sadly, write frame to file, as we need them indexed for the classification during tracks
                # TODO try to find a way without so much read/write to disk?
                count_str = '{:06d}'.format(count)
                image_name = self.video_name + '_frame_' + count_str + self.image_suffix
                
                save_path = os.path.join(self.save_frame_dir, image_name)
                cv.imwrite(save_path, frame)
                
                # track and detect single frame
                box_array = self.GetTracksFromFrame(frame, imgw, imgh)

                det = ImageWithDetection(txt_file='empty',
                                         image_name=save_path,
                                         detection_data=box_array,
                                         image_width=imgw,
                                         image_height=imgh)
                
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
        
        return image_detection_list, imgw, imgh


    def CountPaintedTurtles(self, tracks):
        '''count painted turtle tracks, ignoring the classified overall'''
        painted_count = 0
        unpainted_count = 0
        painted_list = []
        turtle_lsit = []
        for i, track in enumerate(tracks):
            for j in enumerate(track.classifications):
                if j[1] == 0 and track.id not in painted_list:
                    painted_list.append(track.id)
                    painted_count += 1
                elif j[1] == 1 and track.id not in turtle_lsit:
                    turtle_lsit.append(track.id)
                    unpainted_count += 1
                # NOTE will count a flickering turtle as both painted and unpainted
        return painted_count, unpainted_count


    def CountPaintedTurtlesOverall(self, tracks):
        """ count painted turtle tracks, tracks must be classified overall """
        painted_count = 0
        unpainted_count = 0
        for i, track in enumerate(tracks):
            if track.classification_overall:
                painted_count += 1
            else:
                unpainted_count += 1
        return painted_count, unpainted_count
    
    
    def MakeVideoAfterTracks(self, image_detection_track_list, MAX_COUNT=0):
        """ make video of detections after tracks classified, etc """
        
       # TODO skip same # of frames as during the reading
        print('MakeVideoAfterTracks')
        
        # read in the video frames
        vidcap = cv.VideoCapture(self.video_path)
        if not vidcap.isOpened():
            print(f'Error opening video file: {self.video_path}')
        
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
        if MAX_COUNT == 0:
            MAX_COUNT = 1e6
        # iterate over video
        while vidcap.isOpened() and count <= MAX_COUNT:
            success, frame = vidcap.read()
            if not success:
                print('ending video capture: vidcap success = false')
                vidcap.release()
                break
            
            if count % self.frame_skip == 0:
                print(f'writing frame: {count}')
            # apply detections/track info to frame
                #    [class, x1,y1,x2,y2, confidence, track id, classifier class, conf class]
                image_data = image_detection_track_list[count]
                box_array = [image_data.get_detection_track_as_array(i, OVERALL=True) for i in range(len(image_data.detections))]
                
                # make plots
                plotter.boxwithid(box_array, frame)
                
                # save to image writer
                out.write(frame)
                
                # TODO save different box arrays frame by frame (overall = false and overall = true) into different folders so can contrast

            count += 1
            if count == MAX_COUNT:
                print(f'frame MAX_COUNT {count} reached')

        out.release()
        vidcap.release()
    
    
    def Run(self, SHOW=False, MAX_COUNT=10):
        # set up storing varibles
        # P_list, transformed_imglist = [], []
        # MAX_COUNT = 10 - maximum number of frames before reading video stops
        
        start_time = time.time()

        # get detection list for each image
        image_detection_list, image_width, image_height = self.GetTracksFromVideo(SHOW, MAX_COUNT)
        # NOTE: we don't need to save each frame, because each frame is already 
        # just want to save the detections/metadata to a file for replotting
        # and we re-open the video when it's time to make the video with detections/plots
        
        # TODO should be input
        tracker_obj = Tracker(self.video_path, 
                              self.save_dir, # TODO check save_dir vs save_frame_dir
                              classifier_weights=self.classifier_weights, 
                              yolo_dir=self.yolo_path,
                              image_width=image_width, 
                              image_height=image_height)
        
        # convert from image list detections to tracks
        tracks = tracker_obj.convert_images_to_tracks(image_detection_list)
        
        # run classifier on tracks
        # NOTE: requires the actual image!
        tracks_classified = tracker_obj.classify_tracks(tracks)
        
        # run classification overall on classified tracks
        tracks_overall = tracker_obj.classify_tracks_overall(tracks_classified)        
        
        # convert tracks back into image detections!
        image_detection_track_list = tracker_obj.convert_tracks_to_images(tracks_overall,image_width, image_height)
        
        # plot classified tracks to file by re-opening the video and applying our tracks back to the images
        self.MakeVideoAfterTracks(image_detection_track_list, MAX_COUNT)
        
        # for overall counts of painted turtles:
        painted, unpainted = self.CountPaintedTurtlesOverall(tracks_overall)
        print("Overal counts")
        print(f'painted count: {painted}')
        print(f'unpainted count: {unpainted}')
        
        painted, unpainted = self.CountPaintedTurtles(tracks_classified)
        print("Count along the way")
        print(f'painted count: {painted}')
        print(f'unpainted count: {unpainted}')
        
        print('counting complete')
        end_time = time.time()
        sec = end_time - start_time
        print('compute time: {} sec'.format(sec))
        print('compute time: {} min'.format(sec / 60.0))
        print('compute time: {} hrs'.format(sec / 3600.0))
        
        code.interact(local=dict(globals(), **locals()))
            
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
                  'frame_skip': yaml_data['frame_skip']}
        return config


if __name__ == "__main__":
    
    
    config_file = 'pipeline_config.yaml' # locally-referenced
    p = Pipeline(config_file=config_file)
    results = p.Run(MAX_COUNT=1000)
    # txt_name = '/home/dorian/Code/turtles/turtle_datasets/tracking_output/test.txt'
    # p.SaveTurtleTotalCount(txt_name, T_count, P_count)

    # p.MakeVideo('031216amnorth', transformed_imglist)
