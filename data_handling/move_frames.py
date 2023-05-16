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
input_frame_dir = '/home/dorian/Code/turtles/turtle_datasets/081220-0310AMsouth-ME2D'


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

# for the 1000 frames, take 200 total, segments of 20 images
# have a start
# have a stop

# TODO: perhaps STRIDE (every X frames to fulfill 200) would have been better?
idx = np.arange(0, len(framelist))
num_segments = np.round(len(idx) / 20) # because we want segments of 20
frame_split = np.array_split(idx, num_segments)
frames_train = np.concatenate(frame_split[::5]) # take every 5, end result should be a 200 length array

# count_frames_train = np.concatenate((np.arange(start=200, stop=300),
#                                     np.arange(start=450, stop=550),
#                                     np.arange(start=850, stop=950))) 

# move frames into train, the rest into test
print('moving frames')
count = 0
for i, framepath in enumerate(framelist):
    framename = os.path.basename(framepath)
    if i in frames_train:
        count += 1
        print(f'{count}: moving img to train')
        shutil.move(os.path.join(framepath), 
                    os.path.join(train_dir, framename))
    else:
        shutil.move(os.path.join(framepath),
                    os.path.join(test_dir, framename))
    
print('done')

# terminal command to undo the moves:
# mv test/*.jpg . && mv train/*.jpg .