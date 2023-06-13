import cv2
import os
import glob
import code
import numpy as np
from PIL import Image
from ultralytics import YOLO

from classifier.Classifier import Classifier
from plotter.Plotter import Plotter
from tracker.ImageWithDetection import ImageWithDetection
# TODO make better and neaten up
'''Class that takes a video and outputs a video file with tracks and
classifications and a text file with final turtle counts'''

# added for memory profiling, see where growing memory usage is
# from guppy import hpy # conda install -c conda-forge guppy3 or pip install guppy3
# h = hpy()
# import sys

class Pipeline:
    default_vid_in = '/home/dorian/Code/turtles/turtle_datasets/041219-0569AMsouth.mp4'
    default_save_dir = '/home/dorian/Code/turtles/turtle_datasets/tracking_output'
    default_track_model = YOLO(
        '/home/dorian/Code/turtles/turtle_tracker/weights/20230430_yolov8x_turtlesonly_best.pt')
    default_classifier_weight = '/home/dorian/Code/turtles/turtle_tracker/weights/classifier_exp35.pt'
    default_yolo_location = '/home/dorian/Code/turtles/yolov5_turtles'
    image_suffix = '.jpg'
    sf = 0.3 # TODO JAVA - try to be more descriptive for a variable name that looks like it should be tweaked


    def __init__(self,
                 video_in: str = default_vid_in,
                 save_dir: str = default_save_dir,
                 yolo_path: str = default_yolo_location,
                 classifier_weight: str = default_classifier_weight,
                 track_model=default_track_model):
        self.vid_path = video_in
        self.save_dir = save_dir
        self.model_track = track_model
        self.model_track.fuse()
        self.classifier_weights = classifier_weight
        yolo_path = yolo_path
        self.classifier = Classifier(weights_file=self.classifier_weights, 
                                     yolo_dir=yolo_path, 
                                     confidence_threshold=0.8)


    def MakeVideo(self,
                  name_vid_out,
                  transformed_imglist,
                  video_in_location: str = default_vid_in):
        '''Given new video name, a list of transformed frames and a video based
        off, outputs a video'''
        vidcap = cv2.VideoCapture(video_in_location)
        w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # code.interact(local=dict(globals(), **locals()))
        video_out = video_in_location.split('.')[0] + name_vid_out + '.mp4'
        out = cv2.VideoWriter(video_out, 
                              cv2.VideoWriter_fourcc(*"mp4v"), 
                              30, 
                              (w, h), 
                              isColor=True)
        for image in transformed_imglist:
            out.write(image)
        out.release()


    def SaveTurtleTotalCount(self, txt_file, T_count, P_count):
        '''Given turtle and painted turtle count, write txt file with turtle and
        painted turtle count'''
        with open(txt_file, 'w') as f:
            f.write(f'{T_count} {P_count}')
        return T_count, P_count


    def GetTracks(self, frame, imgw, imgh):
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
                                  int(float(xyxyn[0])*imgw),    # x1
                                  int(float(xyxyn[1])*imgh),    # y1
                                  int(float(xyxyn[2])*imgw),    # x2
                                  int(float(xyxyn[3])*imgh),    # y2
                                  float(boxes.conf[i]),         # conf
                                  int(boxes.id[i])])            # track id
        return box_array


    def ClassifynCount(self, frame, box_array, imgw, imgh, T_count,P_list):
        '''Given an image and box information around each turtle, clasify each 
        turtle adding the conf and classification to the box array'''
        for box in box_array:
                xmin, xmax, ymin, ymax = max(0,box[1]), min(
                     box[3],imgw), max(0,box[2]), min(box[4],imgh)
                cls_img_crop = frame[ymin:ymax,xmin:xmax]
                image = Image.fromarray(cls_img_crop)
                pred_class, predictions = self.classifier.classify_image(
                     image)
                if not bool(pred_class): #prediction not made / confidence too low (pred_class is empty)
                    p = 0 #mark as turtle
                else: 
                    p = (int(pred_class[0]))
                box.append(p)
                box.append(1-predictions[p].item())
                id = box[6]
                if id > T_count:
                    T_count = id
                if box[7] == 1 and (id not in P_list): #if painted and unique id
                    P_list.append(id)
                    #code.interact(local=dict(globals(), **locals()))
        return box_array, T_count, P_list


    def Run(self, Show):
        # set up storing varibles
        P_list, transformed_imglist = [], []
        T_count, count, MAX_COUNT = 0, 0, 1000

        cap = cv2.VideoCapture(self.vid_path)
        if not cap.isOpened():
            print(f'Error opening video file: {self.vid_path}')

        while cap.isOpened() and count <= MAX_COUNT:
            success, frame = cap.read()
            if not success:
                break

            print(f'frame: {count}')
            imgw, imgh = frame.shape[1], frame.shape[0]
            plotter = Plotter(imgw, imgh)

            # track and detect
            box_array = self.GetTracks(frame, imgw, imgh)

            print(f'classifing frame {count}')
            img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box_array, T_count, P_list = self.ClassifynCount(
                img2, box_array, imgw, imgh, T_count, P_list)

            plotter.boxwithid(box_array, frame)

            if Show:
                img = cv2.resize(frame, None, fx=self.sf,
                                 fy=self.sf, interpolation=cv2.INTER_AREA)
                cv2.imshow('images', img)
                cv2.waitKey(0)
                print(T_count, len(P_list))
            # code.interact(local=dict(globals(), **locals()))


            # NOTE: we don't need to save each frame, because each frame is already 
            # just want to save the detections/metadata to a file for replotting
            # and we re-open the video when it's time to make the video with detections/plots
            
            # transformed_imglist.append(frame)
            # temporarily commented out - might have to save frames instead of committing them to memory...
            count += 1
            
            # for memory profiling
            # if count == 100:
            #     print(h.heap())
            #     code.interact(local=dict(globals(), **locals()))
            
            
        return transformed_imglist, T_count, len(P_list)


if __name__ == "__main__":
    p = Pipeline()
    
    transformed_imglist, T_count, P_count = p.Run(Show=False)
    txt_name = '/home/dorian/Code/turtles/turtle_datasets/tracking_output/test.txt'
    p.SaveTurtleTotalCount(txt_name, T_count, P_count)

    # p.MakeVideo('031216amnorth', transformed_imglist)