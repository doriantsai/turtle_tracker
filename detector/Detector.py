#! /usr/bin/env/python3

import os
from typing import Tuple
import torch
import cv2 as cv
import numpy as np
import glob

# from ultralytics import YOLO

from plotter.Plotter import Plotter

"""Detector.py
class definition for detector of turtles
"""


class Detector:
    
    YOLO_PATH_DEFAULT = '/home/dorian/Code/turtles/yolov8_turtles'
    WEIGHTS_FILE_DEFAULT = '/home/dorian/Code/turtles/yolov8_turtles/weights/20230430_yolov8x_turtlesonly_best.pt'
    SAVE_DIR_DEFAULT = '/home/dorian/Code/turtles/turtle_tracker/output'
    DETECTION_IMAGE_SIZE_DEFAULT = [1024, 1024]
    CONFIDENCE_THRESHOLD_DEFAULT = 0.5
    IMG_SUFFIX_DEFAULT = '*.PNG'
    # IOU_THRESHOLD_DEFAULT = 0.5
    
    
    def __init__(self,
                 weights_file: str = WEIGHTS_FILE_DEFAULT,
                 yolo_dir: str = YOLO_PATH_DEFAULT,
                 detection_image_size: Tuple[int, int] = DETECTION_IMAGE_SIZE_DEFAULT,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.yolo_dir = yolo_dir
        self.weights_file = weights_file
        
        self.model = self.load_model(weights_file)
        
        self.class_names = self.model.names
        self.model.conf = confidence_threshold
        
        self.detection_image_size = detection_image_size
        # self.original_input_image_size = []
        

    def load_model(self, weights_file: str, local: bool = False):
        """load_model
        load the pytorch weights file (yolov8) for detection

        Args:
            weights_file (str): absolute path to weights_file
        """
        # TODO check that is a valid file  
        if local:
            model = torch.hub.load(self.yolo_dir, 'custom', path=weights_file, source='local')
        else:
            # online, requires network connection
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_file, trust_repo=True)
        # model = YOLO(weights_file)
            
        # model.agnostic = True
        # model.max_det = 1000
        # model.MODE('predict')
        # model.TASK('detect')
        # model.imgsz = 640
        model = model.to(self.device)
        return model
        
        
    def pred2array(self, results):
        """pred2array
        take detector model predictions and convert them to an array

        Args:
            results (list?): detector results, raw

        Returns:
            np_array: array = xmin, xmax, ymin, ymax, confidence, class
        """           
        pred = results.pred[0]
        # pred = results.keypoints
        predarray = []
        if pred is not None:    
            for i in range(len(pred)):
                row = []
                for j in range(6):
                    row.append(pred[i,j].item())
                predarray += (row)    
            predarray = np.array(predarray)
            predarray = predarray.reshape(len(pred),6)
        return predarray


    def read_image(self, image_path: str):
        """read_image
        read in image for detection from image_path, return image as numpy array
        in RGB format

        Args:
            image_path (str): absolute image path to image file
        """
        image = cv.imread(image_path)
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)
        

    def convert_input_image(self, image, image_color_format: str = 'RGB'):
        """convert_input_image

        Args:
            image (numpy array, PIL image or Tensor): input image to give model
            image_color_format (str, optional): determine if numpy array is and
            RGB or BGR format. Defaults to 'RGB'.
        """
        color_format = ['RGB', 'BGR']
        
        if isinstance(image, np.ndarray):
            # image is numpy array
            # re-order channels if BGR
            if image_color_format == color_format[1]:
                # image = image[:, :, [2,1,0]]
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            # TODO check input image size
            # TODO check resize?
            # TODO check format - to tensor or numpy array, PILLOW image?
            # TODO normalize image?
        return image 
        
        
    def detect(self, image):
        """detect
        perform detection on image
        
        Args:
            img (_type_): _description_
        """
        
        results = self.model([image], size=self.detection_image_size)
        # results = self.model.predict(source=image)
        
        # TODO process/refine results?
        return results


    def run(self,
            image_dir: str,
            save_dir: str = SAVE_DIR_DEFAULT,
            annotation_dir: str = None,
            save_imgs: bool = True):
        """run
        run detector over image directory, compare to annotations if available,
        and save images to directory if save_imgs is True

        Args:
            image_dir (str): _description_
            save_dir (str, optional): _description_. Defaults to SAVE_DIR_DEFAULT.
            annotation_dir (str, optional): _description_. Defaults to None.
            save_imgs (bool, optional): _description_. Defaults to True.
        """
        
        # get images
        # NOTE the sorted ensures that image_list and ann_list correspond
        image_list = sorted(glob.glob(os.path.join(image_dir, self.IMG_SUFFIX_DEFAULT)))
        
        if annotation_dir is not None:
            # get annotations/text list
            ann_list = sorted(glob.glob(os.path.join(annotation_dir, '*.txt')))
            
        # iterate through each image
        predictions_list = []
        max_image = 5 # temporary debug
        for i, image_name in enumerate(image_list):
            if i > max_image:
                break
            
            print(f'{i+1}/{len(image_list)}: {os.path.basename(image_name)}')
            
            # read in image
            image = self.read_image(image_name)
            image_width, image_height = image.shape[1], image.shape[0]
            
            # convert image to correct format
            image = self.convert_input_image(image)
        
            # detection
            results = self.detect(image)
            
            predictions = self.pred2array(results)
            
            # initialise plotting object #TODO maybe should be static method?
            PlotTurtles = Plotter(image_width, image_height)
            
            if annotation_dir is not None:
                # plot groundtruth
                PlotTurtles.text2box(ann_list[i], image)
            
            # plot detections
            PlotTurtles.predarray2box(predictions, image)
            
            if save_imgs:
                # save image to file
                PlotTurtles.save_image(image, os.path.join(save_dir, os.path.basename(image_name)), 'RGB')
                
            predictions_list.append(predictions)
            # end image_list iteration
        
        return predictions_list
        

if __name__ == "__main__":
    
    print('Detector.py')
    
    weights_file = '/home/dorian/Code/turtle_tracker/detector/weights/20230406_yolov5_turtlesonly_best.pt'
    yolo_dir = '/home/dorian/Code/yolov5'
    save_dir = '/home/dorian/Code/turtle_tracker/detector/output'
    # initialise the detector
    TurtleDetector = Detector(weights_file=weights_file, yolo_dir=yolo_dir)
    
    image_dir = '/home/dorian/Data/turtle_datasets/job10_041219-0-1000/obj_train_data'
    ann_dir = image_dir
    # run the detector
    pred_list = TurtleDetector.run(image_dir, save_dir=save_dir, annotation_dir=ann_dir)
    
    import code
    code.interact(local=dict(globals(), **locals()))