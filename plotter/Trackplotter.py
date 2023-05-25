import cv2
import os
import glob
import code
from PIL import Image
import numpy as np
from classifier.Classifier import Classifier
from plotter.Plotter import Plotter

##############################################################################################

class Trackplotter:
        sf = 0.3
        classifier = Classifier()

        def __init__(self, data_location):
                self.data_location = data_location

        def MakeVideo(self, video_in_location, name_vid_out, transformed_imglist):
                vidcap = cv2.VideoCapture(video_in_location)
                w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_out = video_in_location.split('.')[0]+name_vid_out+'.mp4'
                out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),30,(w,h),isColor=True)
                for image in transformed_imglist:
                        out.write(image)
                #code.interact(local=dict(globals(), **locals()))
                out.release()

        def MakeFrames(self, imglist,txtlist,data_location, Show):
                transformed_imglist = []
                Maximage = 2
                for i, imgname in enumerate(imglist):# for every tracked image:
                        if i > Maximage:
                                break
                        img = cv2.imread(imgname)
                        imgw, imgh = img.shape[1], img.shape[0]
        
                        plotter = Plotter(imgw, imgh)

                        datalines = plotter.track2box(txtlist[i]) #fetch class and xyxy details
                        for dataline in datalines: #for every turtule
                                cls_img = self.classifier.read_image(imgname)
                                cls_img_crop = self.classifier.crop_image(cls_img,dataline[1:5],1,1) #crop turtle
                                pred_class, predictions = self.classifier.classify_image(cls_img_crop) #classifiy it
                                #add classified details to datalist
                                dataline.append(int(pred_class[0]))
                                dataline.append(1-predictions[pred_class].item()) 
                        #create boxes aroud turtles
                        plotter.boxwithid(datalines,img)

                        if Show:
                                img = cv2.resize(img, None, fx=self.sf, fy=self.sf, interpolation=cv2.INTER_AREA)
                                cv2.imshow('images', img)
                                cv2.waitKey(0)

                        transformed_imglist.append(img) 
                        print(f'{i+1}/{len(imglist)}: image {imgname.replace(data_location,"")} processed')

                return transformed_imglist
        
if __name__ == "__main__":
    
        print('Trackplotter.py')
        base_data_path = '/home/raineai/Turtles/datasets/trim_vid'
        data_location = os.path.join(base_data_path, 'output/041219-0569AMsouth_trim')
        video_in = os.path.join(base_data_path,'041219-0569AMsouth_trim.mp4')
        txtlist = sorted(glob.glob(os.path.join(data_location, '*.txt')))
        imglist = sorted(glob.glob(os.path.join(data_location, '*.jpg')))

        plotTracks = Trackplotter(data_location)
        transformed_imglist = plotTracks.MakeFrames(imglist,txtlist,data_location,Show=False)
        plotTracks.MakeVideo(video_in,'FunctionTest',transformed_imglist)

