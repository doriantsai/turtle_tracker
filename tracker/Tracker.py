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


# load model
# model = YOLO('weights/20230430_yolov8x_turtlesonly_best.pt')

def write_track_detections(txt_file, boxes):
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

def main(vid_path, save_dir):
    print('tracking test')
    # model = YOLO('yolov8l.pt')
    model = YOLO('/home/dorian/Code/turtles/turtle_tracker/weights/20230430_yolov8x_turtlesonly_best.pt')
    model.fuse()
    
    # img_dir = '/home/dorian/Code/turtles/turtle_datasets/job10_mini/frames_0_200'
    # img_list = glob.glob(os.path.join(img_dir, '*.PNG'))
    
    vid_name = os.path.basename(vid_path).rsplit('.', 1)[0]
    os.makedirs(os.path.join(save_dir, vid_name), exist_ok=True)
    
    # running the model directly on the large video file will accumulate results in RAM and potentially cause out-of-memory errors
    # result = model.track(source=vid_path, save=True, persist=True)
    
    # therefore, we have to stream the video
    cap = cv.VideoCapture(vid_path)
    
    if not cap.isOpened():
        print(f'Error opening video file: {vid_path}')    
    
    count = 0
    MAX_COUNT = 5 # for debug purposes
    while cap.isOpened() and count <= MAX_COUNT:
        success, frame = cap.read()
        
        if not success:
            break
        
        print(f'frame: {count}')
        results = model.track(source = frame, stream=True, persist=True, save_txt=True, save_conf=True, save=True, boxes=True)
        
        for r in results:
            boxes = r.boxes
            # at this point, saves image, and txt
            
            # create unique text file for each frame
            count_str = '{:06d}'.format(count)
            txt_file = os.path.join(save_dir, vid_name, vid_name + '_frame_' + count_str + '.txt')
            
            # write detections/tracks to unique text file
            write_track_detections(txt_file, boxes)
            
            
            
            
            
            # code.interact(local=dict(globals(),**locals()))        
            
            # plot images to RGB image
            # save results to text file
            # save results to video file

        count += 1
    

    return results


if __name__ == "__main__":
    vid_path = '/home/dorian/Code/turtles/turtle_datasets/041219-0569AMsouth/041219-0569AMsouth_trim.mp4'
    save_dir = 'output3'    
    main(vid_path, save_dir)
    
# import code
