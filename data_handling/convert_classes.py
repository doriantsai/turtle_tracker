#! /usr/bin/env/python3

"""convert_classes.py
script to convert yolov5 classes to desired class
"""


import os
import glob

# set folder for labels
label_dir = '/home/dorian/Data/turtle_datasets/temp_obj_train_data'

# set desired class label
label_default = '1'

# set output directory
output_dir = '/home/dorian/Data/turtle_datasets/temp_obj_train_data_output'
os.makedirs(output_dir, exist_ok=True)

# iterate through all files in given folder
# label_files = sorted(os.listdir(label_dir))
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
                if line[0] == '0':
                    modified_line = line.replace('0', '1', 1)
                else:
                    modified_line = line
                
                # write to output file
                outfile.write(modified_line)

print('done')

import code
code.interact(local=dict(globals(), **locals()))
