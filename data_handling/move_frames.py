#! /usr/bin/env python3

"""
move_frames

automatically move specified number of frames from one folder, and the remainder to another folder
used for splitting up testing/training image frames for annotation
"""

import os
import numpy as np
import shutil
import glob

# input frame directory
input_frame_dir = '/home/dorian/Code/turtles/turtle_datasets/031216AMsouth'


# create sub-directories: training, testing images
train_dir = os.path.join(input_frame_dir, 'train')
test_dir = os.path.join(input_frame_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# get frame list
framelist = sorted(glob.glob(os.path.join(input_frame_dir, '*.jpg'), recursive=False))
# framelist = sorted(os.listdir(input_frame_dir))

# import code
# code.interact(local=dict(globals(), **locals()))

# set number of frames/frame range to grab from
count_frames_train = np.concatenate((np.arange(start=400, stop=450),
                                    np.arange(start=550, stop=650),
                                    np.arange(start=900, stop=950))) 

# move frames into train, the rest into test
print('moving frames')
for i, framepath in enumerate(framelist):
    framename = os.path.basename(framepath)
    if i in count_frames_train:
        shutil.move(os.path.join(framepath), 
                    os.path.join(train_dir, framename))
    else:
        shutil.move(os.path.join(framepath),
                    os.path.join(test_dir, framename))
    
print('done')

# terminal command to undo the moves:
# mv test/*.jpg . && mv train/*.jpg .