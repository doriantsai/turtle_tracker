#! /usr/bin/env/python3

"""convert_classes.py
script to convert yolov5 classes to desired class
also create symbolic links of images into the same folder (could copy/paste, but issues with memory)
we want this to have all labels turtles for initial turtle detector, but want to
retain the painted/not-painted labels for groundtruth and 
"""


import os
import glob

HOME_DIR = '/home/serena/Data/Turtles/turtle_datasets'
VID_DIR = 'job10_041219-0-1000'
IN_DIR = 'obj_train_data'
IMG_TYPE = '.PNG'
OUT_DIR = 'turtle_labels_only'

# set folder for labels
label_dir = os.path.join(HOME_DIR,VID_DIR,IN_DIR)

# TODO should manage classes via dictionary
# set class label that needs changing
label_to_change = '1'

# set desired class label to change to (output class)
label_default = '0'

# set output directory
output_dir = os.path.join(HOME_DIR,VID_DIR,OUT_DIR)
os.makedirs(output_dir, exist_ok=True)


# create symbolic links of images into output directory
# NOTE separate from the label_files, for modularity
print('creating symlinks of images')
img_files = glob.glob(label_dir + '/*'+IMG_TYPE)
for i, img_file in enumerate(img_files):
    print(f'{i+1}/{len(img_files)}: {os.path.basename(img_file)}')
    link_path = os.path.join(output_dir, os.path.basename(img_file))
    os.symlink(img_file, link_path)


# iterate through all files in given folder
# label_files = sorted(os.listdir(label_dir))
print('converting classes in label text files')
label_files = glob.glob(label_dir + '/*.txt')
for i, label_file in enumerate(label_files):
    print(f'{i+1}/{len(label_files)}: {os.path.basename(label_file)}')
    
    # output file name
    output_file = os.path.basename(label_file)
    # input file
    with open(os.path.join(label_file), 'r') as infile, \
        open(os.path.join(output_dir, output_file), 'w') as outfile:
            # loop through each line of the input file
            for line in infile:
                # replace character if it is not the desired class:
                if line[0] == label_to_change:
                    modified_line = line.replace(label_to_change, label_default, 1)
                else:
                    modified_line = line
                
                # write to output file
                outfile.write(modified_line)

print('done')

# import code
# code.interact(local=dict(globals(), **locals()))
