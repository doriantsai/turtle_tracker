import os
import code
import numpy as np
import glob
import cv2 as cv
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Tuple
import code

from ultralytics import YOLO

"""Classifier.py
class definition for classifier of turtles
"""


class Classifier:
    
    WEIGHTS_FILE_DEFAULT = '/home/dorian/Code/turtles/yolov8_turtles/runs/classify/train7/weights/last.pt' 
    # YOLO_PATH_DEFAULT = '/home/dorian/Code/turtles/yolov8_turtles'
    CLASSIFIER_IMAGE_SIZE_DEFAULT = [64, 64]
    CONFIDENCE_THRESHOLD_DEFAULT = 0.5
    IMG_SUFFIX_DEFAULT = '*.jpg'    
    IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
    IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation
    
    
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
        
        self.resize = T.Resize(self.classify_image_size)  
        self.to_tensor = T.ToTensor()  
        self.normalise_img = T.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)


    def load_model(self, weights_file: str):
        """load_model
        load the pytorch weights file (yolov5) for classification

        Args:
            weights_file (str): absolute path to weights_file
        """
        # model = YOLO('yolov8x-cls.pt') # workaround to get model to load properly
        model = YOLO(weights_file)
        return model


    def transform_img(self, img):
        """transform_img

        Args:
            image (numpy array, PIL image or Tensor): input image to give model
        """
        
        img = self.resize(img) # resize from PIL image - works
        img_t = self.to_tensor(img)
        # img_t = self.resize(img_t) # resize from tensor - investigating if works
        # convert to tensor
        img_b = img_t.unsqueeze(0)
        img_b = self.normalise_img(img_b)
        # img_b = img_b.half() if self.model.fp16 else img_b.float() # uint18 to fp16/32
        img_b = img_b.float()
        img_b = img_b.to(self.device)
        return img_b
    
    
    def read_image(self, image_path: str):
        """read_image
        read in image for classification from image_path, return image as numpy array
        in RGB format

        Args:
            image_path (str): absolute image path to image file
        """
        # from PIL image, works
        img = Image.open(image_path).convert('RGB')
        
        # TODO from numpy array - testing
        # img = cv.imread(image_path)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
        return img
    
    
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
        probs = probs.cpu()
        # probs = F.softmax(probs.cpu(), dim=-1) 
        p = probs[0]

        return p
    
        
    def classify_image(self, image):
        """
        # NOTE: upgrade to yolov8, should no longer require PIL
        run classifier on single (cropped) image (PIL) RGB image
        output:
        p = discrete class prediction
        predictions = the pseudo-probabilities output at the end of the classifier
        """
        predictions = self.classify(image)
        
        # show image:
        # import matplotlib.pyplot as plt
    
        # plt.imshow(image)
        # plt.show()
        # code.interact(local=dict(globals(), **locals()))
            
        # print(f'predictions: {predictions}')
        if predictions[0] > self.model.conf:
            p = 0
            conf = predictions[0]
        else:
            p = 1
            conf = predictions[1]
        return p, conf
    
    
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
