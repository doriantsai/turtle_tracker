import cv2
import os
import glob
import code
from classifier.Classifier import Classifier
from plotter.Plotter import Plotter

##############################################################################################

class PlotTracks:
        sf = 0.3
        WEIGHTS_FILE_DEFAULT = '/home/raineai/Turtles/yolov5_turtles/runs/train-cls/exp29/weights/best.pt' 
        YOLO_PATH_DEFAULT = '/home/raineai/Turtles/yolov5_turtles'
        classifier = Classifier(weights_file=WEIGHTS_FILE_DEFAULT, yolo_dir=YOLO_PATH_DEFAULT, confidence_threshold=0.8)
        DEFAULT_DATA_LOCATION = '/home/raineai/Turtles/datasets/trim_vid/output/041219-0569AMsouth_trim'
        DEFAULT_VIDEO_IN = '/home/raineai/Turtles/datasets/trim_vid/041219-0569AMsouth_trim.mp4'

        def __init__(self, 
                     data_location: str = DEFAULT_DATA_LOCATION):
                self.data_location = data_location

        def MakeVideo(self, 
                      name_vid_out, 
                      transformed_imglist,
                      video_in_location: str = DEFAULT_VIDEO_IN):
                '''Given new video name, a list of transformed frames and a video based off make a video'''
                vidcap = cv2.VideoCapture(video_in_location)
                w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_out = video_in_location.split('.')[0]+name_vid_out+'.mp4'
                out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),30,(w,h),isColor=True)
                for image in transformed_imglist:
                        out.write(image)
                out.release()

        def SaveTurtleCount(self, txt_file, data):
                '''Given track information for one image, write txt file with turtle and painted turtle count'''
                T_count, P_list = 0, []
                for d in data:
                        id = d[6]
                        if id > T_count:
                                T_count = id
                        if d[7] == 1 and (id not in P_list): #if painted and unique id
                                P_list.append(id)
                with open(txt_file, 'w') as f:
                        f.write(f'{T_count} {len(P_list)}')
                return T_count, P_list
        
        def SaveTurtleTotalCount(self, txt_file, T_count, P_count):
                '''Given turtle and painted turtle count, write txt file with turtle and painted turtle count'''
                with open(txt_file, 'w') as f:
                        f.write(f'{T_count} {P_count}')
                return T_count, P_count

        def ClassifyWithTrack(self, datalines, T_count, P_list, imgname):
                '''for each track in the image, classify every turtle'''
                for dataline in datalines: #for every turtule
                        cls_img = self.classifier.read_image(imgname)
                        cls_img_crop = self.classifier.crop_image(cls_img,dataline[1:5],1,1) #crop turtle
                        pred_class, predictions = self.classifier.classify_image(cls_img_crop) #classifiy it
                        #add classified details to datalist
                        dataline.append(int(pred_class[0]))
                        dataline.append(1-predictions[pred_class].item()) 
                        id = dataline[6]
                        if id > T_count:
                                T_count = id
                        if dataline[7] == 1 and (id not in P_list): #if painted and unique id
                                P_list.append(id)
                return datalines, T_count, P_list

        def Run(self, 
                       imglist,
                       txtlist,
                       data_location: str = DEFAULT_DATA_LOCATION, 
                       Show: bool = False,
                       Save: bool = False):
                transformed_imglist, P_list, T_count = [], [], 0
                Maximage = 3
                for i, imgname in enumerate(imglist):# for every tracked image:
                        if i > Maximage:
                                break
                        img = cv2.imread(imgname)
                        imgw, imgh = img.shape[1], img.shape[0]
                        plotter = Plotter(imgw, imgh)

                        datalines = plotter.track2box(txtlist[i]) #fetch class and xyxy details
                        datalines, T_count, P_list = self.ClassifyWithTrack(datalines, T_count, P_list, imgname)
                        #create boxes aroud turtles
                        plotter.boxwithid(datalines,img)

                        if Show:
                                img = cv2.resize(img, None, fx=self.sf, fy=self.sf, interpolation=cv2.INTER_AREA)
                                cv2.imshow('images', img)
                                cv2.waitKey(0)
                        if Save:
                                txt_name = '/home/raineai/Turtles/datasets/trim_vid/output/'+imgname.replace(data_location+'/',"")[:-4]+'.txt'
                                self.SaveTurtleCount(txt_name, datalines)   
                        
                        transformed_imglist.append(img) 
                        print(f'{i+1}/{len(imglist)}: image {imgname.replace(data_location,"")} processed')
                        print(f'Total Turtle count is: {T_count}. Total painted turtle count is: {len(P_list)}')

                txt_name = '/home/raineai/Turtles/datasets/trim_vid/output/final.txt'
                self.SaveTurtleTotalCount(txt_name, T_count, len(P_list))
                return transformed_imglist
        
if __name__ == "__main__":
    
        print('TrackPlot.py')
        base_data_path = '/home/raineai/Turtles/datasets/trim_vid'
        data_location = os.path.join(base_data_path, 'output/041219-0569AMsouth_trim')
        video_in = os.path.join(base_data_path,'041219-0569AMsouth_trim.mp4')
        txtlist = sorted(glob.glob(os.path.join(data_location, '*.txt')))
        imglist = sorted(glob.glob(os.path.join(data_location, '*.jpg')))

        plotTracks = PlotTracks(data_location)
        transformed_imglist = plotTracks.Run(imglist,txtlist,data_location,Show=True,Save=False)
        #plotTracks.MakeVideo('FunctionTest',transformed_imglist,video_in)
        

        code.interact(local=dict(globals(), **locals()))

