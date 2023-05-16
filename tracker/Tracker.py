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

# load model
# model = YOLO('weights/20230430_yolov8x_turtlesonly_best.pt')


def main():
    print('tracking test')
    # model = YOLO('yolov8l.pt')
    model = YOLO('/home/dorian/Code/yolov8_tracking_turtles/weights/20230430_yolov8x_turtlesonly_best.pt')
    model.fuse()
    
    img_dir = '/home/dorian/Data/turtle_datasets/job10_mini/obj_train_data'
    img_list = glob.glob(os.path.join(img_dir, '*.PNG'))

    result = model.track(source=img_list, save=True)
    


if __name__ == "__main__":
    main()
    
# import code