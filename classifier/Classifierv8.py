import os
import code
import numpy as np
import glob
from PIL import Image
import torch
from typing import Tuple
import code

from ultralytics import YOLO

"""Classifier.py
class definition for classifier of turtles
"""


class Classifier:

    WEIGHTS_FILE_DEFAULT = '/home/dorian/Code/turtles/turtle_tracker/weights/20230820_yolov8s-cls_best.pt'
    # YOLO_PATH_DEFAULT = '/home/dorian/Code/turtles/yolov8_turtles'
    CLASSIFIER_IMAGE_SIZE_DEFAULT = [64, 64]
    CONFIDENCE_THRESHOLD_DEFAULT = 0.5
    IMG_SUFFIX_DEFAULT = '*.jpg'    
    IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
    IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation
    # painted/unpainted dictionary:
    label_definitions = {'unpainted': '0',
                         'painted': '1'}
    
    def __init__(self,
                 weights_file: str = WEIGHTS_FILE_DEFAULT,
                #  yolo_dir: str = YOLO_PATH_DEFAULT,
                 classifier_image_size: Tuple[int, int] = CLASSIFIER_IMAGE_SIZE_DEFAULT,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.yolo_dir = yolo_dir
        self.weights_file = weights_file
        self.classify_image_size = classifier_image_size
        self.model = self.load_model(weights_file)
        self.class_names = self.model.names
        self.model.conf = confidence_threshold
        

    def load_model(self, weights_file: str):
        """load_model
        load the pytorch weights file (yolov5) for classification

        Args:
            weights_file (str): absolute path to weights_file
        """
        # model = YOLO('yolov8x-cls.pt') # workaround to get model to load properly
        model = YOLO(os.path.expanduser(weights_file))
        return model

    
    
    def classify(self, image):
        """classify
        perform classify on image
        
        Args:
            img (_type_): _description_
        """
        results = self.model(image, verbose=False)
        
        # print("results", results[0].probs)
        probs = results[0].probs.data
        # NOTE: last layer of model is just a linear layer, 
        # so might justify applying softmax at end, 
        # although the results appear to sum to one consistently for Yolov8
        # over predicted results
        probs = probs.cpu().numpy()
        # probs = F.softmax(probs.cpu(), dim=-1) 
        # probs = probs.numpy()
        sum_of_probs = probs[0]+probs[1]
        if abs(1-sum_of_probs) > 0.001:
            print(probs)
        p = probs[0]

        return p
    
        
    
    
    def crop_image(self, image: Image, box):
        """ crop image given bounding box
        xyxyn and original image bounds, PIL image"""
        (image_width, image_height) = image.size
        # print(image_width, image_height)
        # take smaller bounding box
        xmin = int(np.ceil(box.left * image_width))
        ymin = int(np.ceil(box.top * image_height)) 
        xmax = int(np.floor(box.right * image_width)) 
        ymax = int(np.floor(box.bottom * image_height))
        # return image[ymin:ymax, xmin:xmax] # numpy array
        return image.crop((xmin, ymin, xmax, ymax)) # this step is a PIL image function
        
    
    def run(self,
            image_dir: str):
        """run
        run classifier over image directory,
        and returns the predictions

        Args:
            image_dir (str): _description_
        """
    
        image_list = sorted(glob.glob(os.path.join(image_dir, self.IMG_SUFFIX_DEFAULT)))
        # max_image = 10 # temporary debug
        pred_list = {'pred_class': [],
                     'pred_class_name': [],
                     'conf': []}
        
        for i, image_name in enumerate(image_list):
            # if i > max_image:
            #     break
            
            print(f'{i+1}/{len(image_list)}: {os.path.basename(image_name)}')
            
            image = self.read_image(image_name)
            p, conf = self.classify_image(image)
            
            pred_list['pred_class'].append(p)
            pred_list['pred_class_name'].append(self.class_names[p])
            pred_list['conf'].append(conf)
            
        return pred_list
    
    
if __name__ == "__main__":
    
    print('Classifierv8.py')

    # weights_file = '/home/dorian/Code/turtles/turtle_tracker/weights/20230814_yolov8x_classifier.pt'
    weights_file = '/home/dorian/Code/turtles/turtle_tracker/weights/20230820_yolov8s-cls_best.pt'
    image_dir = '/home/dorian/Code/turtles/turtle_datasets/classification/default_train_images'
    
    # initialise the classifier
    ClassyTurtle = Classifier(weights_file = weights_file)
    
    # run the classifier
    print('running classifier')
    pred_list = ClassyTurtle.run(image_dir)
    
    print(pred_list)
    
    
    code.interact(local=dict(globals(), **locals()))
