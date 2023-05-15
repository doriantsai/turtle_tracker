#! /usr/bin/env/python3

""" classify_turtles.py
After training a turtle classifier, do model inference (classify turtles given set of images)
"""

import os
import code
import numpy as np
import glob
import cv2 as cv
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T


# model from training classifier
weights = '/home/dorian/Code/yolov5/runs/train-cls/exp4/weights/best.pt'

# image path
img_dir = '/home/dorian/Data/turtle_datasets/job10_classification/test/turtle'
img_list = sorted(glob.glob(os.path.join(img_dir, '*.PNG')))

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'weights file = {weights}')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
model = model.to(device)
model.eval()

# class definitions
# object_classes = {'turtle': 0,
#                   'painted': 1}
names = model.names

# transforms for image handling
imgsz = [224, 224]
resize = T.Resize(imgsz)  
to_tensor = T.ToTensor()  
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation
normalise_img = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

# model.warmup?
bs = 1
model.warmup(imgsz=(1 if model.pt else bs, 3, *imgsz))
# iterate over each image
for i, img_name in enumerate(img_list):
    
    if i > 10:
        break
    # print(f'image = {img_name}')
    
    img = Image.open(img_name).convert('RGB')
    
    # resize image for classifier:
    img = resize(img)
    
    # convert to tensor
    img_t = to_tensor(img)
    img_b = img_t.unsqueeze(0)
    img_b = normalise_img(img_b)
    img_b = img_b.half() if model.fp16 else img_b.float() # uint18 to fp16/32
    img_b = img_b.to(device)
    
    
    # inference
    results = model(img_b)
    print(f'results = {results}')
    
    # code.interact(local=dict(globals(),**locals()))    
    pred = F.softmax(results, dim=1)
    
    print(f'pred = {pred}')
    
print('done')
code.interact(local=dict(globals(),**locals()))    

