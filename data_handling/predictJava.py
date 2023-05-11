import torch
import code
import os
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import glob
from PIL import Image
import cv2

basepath = os.path.join('/home/raineai/Turtles')
weights = os.path.join(basepath, 'yolov5_turtles/runs/train-cls/exp24/weights/best.pt')
imgpath = os.path.join(basepath,'datasets/job10_2clases/val')
        #/frame_0000072.PNG')
#imgpath = os.path.join(basepath,'datasets/job10_2clases/train/turtle/frame_0000103.PNG')
yolopath = os.path.join(basepath, 'yolov5_turtles')
Conf_T = 0.80 #70 recall=100, 75 recall=98, 80 r=0.94, 85 r=79, 90 r=0.05, 95 r=0

imglistp = sorted(glob.glob(os.path.join(imgpath,'painted', '*.PNG')))
imglistt = sorted(glob.glob(os.path.join(imgpath, 'turtle', '*.PNG')))
imglist = imglistp+imglistt
predlist = []

#load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'weights file = {weights}')
model = torch.hub.load(yolopath, 'custom', path=weights, source='local')
model = model.to(device)
model.eval()
#model.conf = 0.25
#model.iou = 0.45
#model.max_det = 1000
count = 0


for i, imgname in enumerate(imglist):
    if i>3:
        break
    count = count+1
    img = Image.open(imgname).convert('RGB')
    #imshow(img)
    
    if imgname.find('painted')!=-1:
        true_label = 'painted'
    else:
        true_label = 'turtle'
    
    #Resize the image for the model, method 1
    resize = T.Resize([224,224])
    img = resize(img)
    to_tensor = T.ToTensor()
    tensor = to_tensor(img)
    tensor = tensor.unsqueeze(0)
    #print(tensor.shape)
    #img.show()
    
    #resize the image for the model, mthod 2
    #img = cv2.imread(imgname)
    #im = torch.Tensor(img).to(model.device) #make tensor for classification?
    #print(f'Tensor shape = {im.shape}')
    #im = torch.transpose(im, 2,0)
    #if len(im.shape)==3:
     #   im = im[None]
    #results = model(im)

    results = model(tensor)
    pred = F.softmax(results,dim=1)

    print(true_label)

    for x, prob in enumerate(pred):
        top5i = prob.argsort(0,descending=True)[:5].tolist()
        for j in top5i:
            if prob[j]>Conf_T:
                predlist.append(j)
        text = '\n'.join(f'{prob[j]:.2f} {model.names[j]}' for j in top5i)  
        print(text)

#print(predlist)
print(Conf_T)

recall = len(predlist)/count
print(recall)

#precisions of turtles
precission = predlist.count(1)/len(imglistt)
print(precission)


#old, didn't work great
#image needs to be reshaped?
#im = torch.Tensor(img).to(model.device) #make tensor for classification?
#print(f'Tensor shape = {im.shape}')
#im = torch.transpose(im, 2,0)
#if len(im.shape)==3:
#    im = im[None]
#print(f'Tensor shape = {im.shape}')
