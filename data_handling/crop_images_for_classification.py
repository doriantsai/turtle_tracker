#! /usr/bin/env python3

import os
import glob
import code
import cv2 as cv
import numpy as np
import random

# purpose: crop for balanced classification dataset from cvat annotated frames

# setup folder of object detection images and annotations
# read in from CVAT annotations (YOLO format)
# for each annotation file:
#   find corresponding image
#   for each bounding box inside
#   get bounding box - crop image and save
#   if painted: 
#       increment painted count
#       save in painted_dir
#   if unpainted:
#       save in unpainted_dir
#       increment unpainted count

# balance painted/unpainted randomly
# (make new folder with symlinks or just write anew for easier copy/paste later)

# ensure even numbers from multiple videos

def clear_files_in_folder(folder_path):
    """
    delete all files in folder just in case re-runs of this script leave an accumulation of files
    """
    print(f'clearing files in folder: {folder_path}')
    
    # get list of all files
    file_list = os.listdir(folder_path)
    
    # loop through each file and delete
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Deleted: {file_name}")
            else:
                print(f"Skipped: {file_name} (not a file)")
        except Exception as e:
            print(f"Error deleting {file_name}: {e}")
    
    # TODO once confirmed working, reduce print statements

def read_yolo_labels(file_path):
    """ read yolo labels given a filepath, output as a list of label dictionaries """
    # also, converts the strings to floats for the coordinates - label is still string
    
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            # label
            # x_center_norm
            # y_center_norm
            # box_width_norm
            # box_height_norm
            lbl, xcn, ycn, bwn, bhn = line.split()
            
            # put into a dictionary for sorting - mainly just want painted turtles
            label = {
                'class': lbl,
                'x_center_norm': float(xcn),
                'y_center_norm': float(ycn),
                'box_width_norm': float(bwn),
                'box_height_norm': float(bhn)
            }
            labels.append(label)
    return labels


def convert_normalised_box_to_pixel_box(label, img_width, img_height, pad):
    """
    convert normalised box to pixels in [xmin, ymin, xmax, ymax] format (numpy array)
    also adds padding around image crop
    also accounts for negatives/image borders because of padding
    """    
    # convert from normalised values to pixels
    x_center = label['x_center_norm'] * img_width
    y_center = label['y_center_norm'] * img_height
    box_width = label['box_width_norm'] * img_width
    box_height = label['box_height_norm'] * img_height
    
    # create minimum/maximum bounding points for box, also add padding
    xmin = round(x_center - box_width / 2) - pad
    xmax = round(x_center + box_width / 2) + pad
    ymin = round(y_center - box_height / 2) - pad
    ymax = round(y_center + box_height / 2) + pad
    
    # need to avoid negatives/image borders
    if xmin <= 0:
        xmin = 1
    if xmax > img_width:
        xmax = img_width
    if ymin <= 0:
        ymin = 1
    if ymax > img_height:
        ymax = img_height
    
    return xmin, ymin, xmax, ymax


# root directory of a given dataset
root_dir = '/home/dorian/Code/turtles/turtle_datasets/job10_mini'

# just in case images/labels are in different directories (currently not)
img_dir = os.path.join(root_dir, 'frames_0_200')
lbl_dir = os.path.join(root_dir, 'frames_0_200')
print(f'grabbing images from {img_dir}')
print(f'grabbing labels from {lbl_dir}')

# painted/unpainted dictionary:
label_definitions = {'unpainted': '0',
                     'painted': '1'}

# grab list of all images/label files (YOLO)
img_list = sorted(glob.glob(os.path.join(img_dir, '*.PNG')))
lbl_list = sorted(glob.glob(os.path.join(lbl_dir, '*.txt')))
print(f'number of images: {len(img_list)}')
print(f'number of labels: {len(lbl_list)}')

# quick sanity check, assuming if == then we have all images/label pairs
if len(img_list) != len(lbl_list):
    print('ERROR: number of images is not equal to number of labels')
    exit()
    
# saving output
out_dir = os.path.join(root_dir, 'classification')
out_dir_painted = os.path.join(out_dir, 'painted_turtles')
out_dir_unpainted = os.path.join(out_dir, 'unpainted_turtles')

os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir_painted, exist_ok=True)
os.makedirs(out_dir_unpainted, exist_ok=True)

# clear folders after each run just to make sure we don't accummulate files over successive runs
clear_files_in_folder(out_dir_painted)
clear_files_in_folder(out_dir_unpainted)

# counters
n_painted_dataset = 0
n_unpainted_dataset = 0

# padding for image crops:
pad = 5

# main loop: iterate over lbl_list:
for i, lbl_file in enumerate(lbl_list):
    
    
    # assuming images/labels are sorted, they have the same ordering
    img_file = img_list[i]
    img_name = os.path.basename(img_file).split('.')[0]
    
    
    
    img = cv.imread(img_file)
    img_width, img_height = img.shape[1], img.shape[0]
    
    # https://docs.ultralytics.com/datasets/detect/
    # format of label file:
    # [class, x_center_n,  y_center_n, width_n, height_n]
    
    # read labels from lbl file
    labels = read_yolo_labels(lbl_file)
    
    # find all painted turtles
    # count how many there are, try to find equivalent number of unpainted turtles
    # n_painted_img = 0
    painted_labels = [label for label in labels if label['class'] == label_definitions['painted']]
    n_painted_img = len(painted_labels)
    
    for label in painted_labels:
        # if label['class'] == label_definitions['painted']:
            
            # for painted turtle, crop image
            # convert label into pixel coordinates
            xmin, ymin, xmax, ymax = convert_normalised_box_to_pixel_box(label, img_width, img_height, pad)
            
            # crop image based on box
            img_crop = img[ymin:ymax, xmin:xmax, :]
            
            img_crop_name = img_name + '_painted_crop_' + str(n_painted_dataset).zfill(4) + '.jpg'
            
            
            # write image
            cv.imwrite(os.path.join(out_dir_painted, img_crop_name), img_crop)
            
            # n_painted_img += 1
            # increment whole-dataset counter
            n_painted_dataset += 1
    
    
    # referencing list of dictionary items, we try to randomly get an equal number of unpainted turtles
    # NOTE: in the unlikely case of more painted than unpainted turtles in image, we simply stop
    # minor class imbalance acceptable
        
    # list comprehension for all unpainted labels
    unpainted_labels = [label for label in labels if label['class'] == label_definitions['unpainted']]
    n_unpainted_img = len(unpainted_labels)
    
    if n_painted_img > 0 and n_unpainted_img >= n_painted_img:
        # randomly select from unpainted_labels
        unpainted_selection = random.sample(unpainted_labels, n_painted_img)
        
        for label in unpainted_selection:
            xmin, ymin, xmax, ymax = convert_normalised_box_to_pixel_box(label, img_width, img_height, pad)
            # crop image based on box
            img_crop = img[ymin:ymax, xmin:xmax, :]
            img_crop_name = img_name + '_unpainted_crop_' + str(n_unpainted_dataset).zfill(4) + '.jpg'
            
            # write image
            cv.imwrite(os.path.join(out_dir_unpainted, img_crop_name), img_crop)
            # increment whole-dataset counter
            n_unpainted_dataset += 1
            
    print(f'frame: {i}: {img_name}, painted: {n_painted_img}')
    
print(f'number of painted turtles: {n_painted_dataset}')
print(f'number of unpainted turtles randomly selected: {n_unpainted_dataset}')
             




print('done')

code.interact(local=dict(globals(),**locals()))




