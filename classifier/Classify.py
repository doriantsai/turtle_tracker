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

"""Classify.py
class definition for classifier of turtles
"""

class Classify:
    BASE_PATH_DEFAULT = '/home/raineai/Turtles'
    WEIGHTS_FILE_DEFAULT = '/home/raineai/Turtles/yolov5_turtles/runs/train-cls/exp26/weights/best.pt' 
    YOLO_PATH_DEFAULT = '/home/raineai/Turtles/yolov5_turtles'
    CLASSIFY_IMAGE_SIZE_DEFAULT = [224, 224]
    CONFIDENCE_THRESHOLD_DEFAULT = 0.5
    IMG_SUFFIX_DEFAULT = '*.PNG'    
    IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
    IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation
    
    def __init__(self,
                 weights_file: str = WEIGHTS_FILE_DEFAULT,
                 yolo_dir: str = YOLO_PATH_DEFAULT,
                 classify_image_size: Tuple[int, int] = CLASSIFY_IMAGE_SIZE_DEFAULT,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
                 imgnet_mean: float = IMAGENET_MEAN,
                 imgnet_std: float = IMAGENET_STD ):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.yolo_dir = yolo_dir
        self.weights_file = weights_file
        self.classify_image_size = classify_image_size
        self.model = self.load_model(weights_file, True)
        self.class_names = self.model.names
        self.model.conf = confidence_threshold
        self.resize = T.Resize(self.classify_image_size)  
        self.to_tensor = T.ToTensor()  
        self.normalise_img = T.Normalize(imgnet_mean, imgnet_std)

    def load_model(self, weights_file: str, local: bool = False):
        """load_model
        load the pytorch weights file (yolov5) for classification

        Args:
            weights_file (str): absolute path to weights_file
        """
        if local:
            model = torch.hub.load(self.yolo_dir, 'custom', path=weights_file, source='local')
        else:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_file, trust_repo=True)
        model = model.to(self.device)
        model.eval()
        model.warmup(imgsz=(1 if model.pt else 1, 3, *self.classify_image_size))
        return model

    def transform_img(self, img):
        """transform_img

        Args:
            image (numpy array, PIL image or Tensor): input image to give model
        """
        img = self.resize(img)
        # convert to tensor
        img_t = self.to_tensor(img)
        img_b = img_t.unsqueeze(0)
        img_b = self.normalise_img(img_b)
        img_b = img_b.half() if self.model.fp16 else img_b.float() # uint18 to fp16/32
        img_b = img_b.to(self.device)
        return img_b
    
    def read_image(self, image_path: str):
        """read_image
        read in image for classification from image_path, return image as numpy array
        in RGB format

        Args:
            image_path (str): absolute image path to image file
        """
        img = Image.open(image_path).convert('RGB')
        return img
    
    def classify(self, image):
        """classify
        perform classify on image
        
        Args:
            img (_type_): _description_
        """
        results = self.model(image)
        pred = F.softmax(results, dim=1)
        return pred
    
    def run(self,
            image_dir: str):
        """run
        run classifier over image directory,
        and returns the predictions

        Args:
            image_dir (str): _description_
        """
        image_list_painted = sorted(glob.glob(os.path.join(image_dir,self.class_names[0], self.IMG_SUFFIX_DEFAULT)))
        image_list_turtles = sorted(glob.glob(os.path.join(image_dir,self.class_names[1], self.IMG_SUFFIX_DEFAULT)))
        image_list = image_list_painted+image_list_turtles
        #image_list = sorted(glob.glob(os.path.join(image_dir, self.IMG_SUFFIX_DEFAULT)))
        predlist = []
        max_image = 100 # temporary debug
        for i, image_name in enumerate(image_list):
            #if i > max_image:
                #break
            print(f'{i+1}/{len(image_list)}: {os.path.basename(image_name)}')
            image = self.read_image(image_name)
            image_transformed = self.transform_img(image)
            predictions = self.classify(image_transformed)

            for x, prob in enumerate(predictions):
                top5i = prob.argsort(0,descending=True)[:5].tolist()
                for j in top5i:
                    if prob[j]>self.CONFIDENCE_THRESHOLD_DEFAULT:
                        predlist.append((j+1) % 2)
        return predlist
    
if __name__ == "__main__":
    
    print('Classify.py')
    
    # initialise the classifier
    TurtleDetector = Classify()
    
    image_dir = os.path.join('/home/raineai/Turtles','datasets/job10_2clases/val')
    
    # run the classifier
    pred_list = TurtleDetector.run(image_dir)
    
    print(pred_list)
    import code
    code.interact(local=dict(globals(), **locals()))