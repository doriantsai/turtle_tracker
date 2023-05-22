#! /usr/bin/env python3

"""Tracker.py
class definition for tracker of turtles
"""

# TODO 
# make sure track_turtles.py works programmatically
# init
# load_model
# accept input from Detector (detections)
# accept input from path of text files (detections)
# run function
# main


from ultralytics import YOLO
import os
import glob
import cv2 as cv
import code
import numpy as np

from tracker.ImageWithDetection import ImageWithDetection

# load model
# model = YOLO('weights/20230430_yolov8x_turtlesonly_best.pt')

class Tracker():
    
    def __init__(self, video_file: str, save_dir: str):
        self.video_file = video_file # TODO can remove vid_path from function input
        self.save_dir = save_dir
        
        self.vid_name = os.path.basename(self.video_file).rsplit('.', 1)[0]
        
        
        
    def write_track_detections(self, txt_file, boxes):
        """
        write yolov8 box to text file
        """
        
        # open text file
        with open(txt_file, 'w') as f:
            for i, id in enumerate(boxes.id):
                
                # class x1 y1 x2 y2 (normalised) conf id
                xyxyn = np.array(boxes.xyxyn[i,:])
                f.write(f'{int(boxes.cls[i])} {xyxyn[0]:.6f} {xyxyn[1]:.6f} {xyxyn[2]:.6f} {xyxyn[3]:.6f} {boxes.conf[i]:.4f} {int(boxes.id[i])}\n')

        return True
        
    # TODO plot tracks to RGB images
    # do this by making new functions in Plotter, leveraging box plots and confidence plots, but simply adding the tracking ID
    # show the track from previous frames? This is secondary


    def read_tracks_from_file(self, txt_dir, txt_search_pattern = '*.txt'):
        """
        read tracks from a text directory and return list of files and their detections and image names
        """
        
        txt_files = glob.glob(os.path.join(txt_dir, txt_search_pattern))
        
        image_list = []
        for txt in txt_files:
            # read textfile:
            # with open(txt, 'r') as f:
            #     f.readlines()
            data = np.loadtxt(txt, dtype=float)
            
            # create ImageWithDetection object
            image_name = os.path.basename(txt).rsplit('.', 1)[0]
            det = ImageWithDetection(txt, image_name=image_name, detections=data)
            
            # TODO maybe use PIL to grab image height/width and populate ImageWithDetection properties? should be redundant though when we open up the image later on anyways
            
            image_list.append(det)
        return image_list

    
    def get_tracks_from_video(self, save_dir):
        print('tracking test')
        # model = YOLO('yolov8l.pt')
        model = YOLO('/home/dorian/Code/turtles/turtle_tracker/weights/20230430_yolov8x_turtlesonly_best.pt')
        model.fuse()
        
        # img_dir = '/home/dorian/Code/turtles/turtle_datasets/job10_mini/frames_0_200'
        # img_list = glob.glob(os.path.join(img_dir, '*.PNG'))
        
        # running the model directly on the large video file will accumulate results in RAM and potentially cause out-of-memory errors
        # result = model.track(source=vid_path, save=True, persist=True)
        
        # therefore, we have to stream the video
        cap = cv.VideoCapture(self.video_file)
        
        if not cap.isOpened():
            print(f'Error opening video file: {self.video_file}')    
        
        count = 0
        MAX_COUNT = 5 # for debug purposes
        while cap.isOpened() and count <= MAX_COUNT:
            success, frame = cap.read()
            
            if not success:
                break
            
            print(f'frame: {count}')
            results = model.track(source = frame, stream=True, persist=True, boxes=True)
            
            for r in results:
                boxes = r.boxes
                # at this point, saves image, and txt
                
                # create unique text file for each frame
                count_str = '{:06d}'.format(count)
                txt_file = os.path.join(save_dir, self.vid_name + '_frame_' + count_str + '.txt')
                
                # write detections/tracks to unique text file
                self.write_track_detections(txt_file, boxes)
                
            count += 1

        return results


    def main(self):
        
        save_txt_dir = os.path.join(self.save_dir, self.vid_name)
        os.makedirs(save_txt_dir, exist_ok=True)
        
        # get tracks into file
        self.get_tracks_from_video(save_txt_dir)
        
        # read_tracks_from_file
        image_list = self.read_tracks_from_file(txt_dir=save_txt_dir)

        # convert image list to tracks



if __name__ == "__main__":
    vid_path = '/home/dorian/Code/turtles/turtle_datasets/041219-0569AMsouth/041219-0569AMsouth_trim.mp4'
    save_dir = 'output3'  
    track = Tracker(vid_path, save_dir)  
    track.main()

    
# import code
