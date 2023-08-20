#!/bin/bash

# bash script to run crop_images_for_classification.py over several videos/folders
# make sure to chmod +x bash_crop_for_class.sh to run in terminal
# ./bash_crop_for_class.sh

python crop_images_for_classification.py --root_dir  '/home/dorian/Code/turtles/turtle_datasets/041219-0569AMsouth/frames_0to1000'
python crop_images_for_classification.py --root_dir  '/home/dorian/Code/turtles/turtle_datasets/041219-0569AMsouth/frames_1001to2000'
python crop_images_for_classification.py --root_dir  '/home/dorian/Code/turtles/turtle_datasets/041219-0569AMsouth/frames_2001to3000'
python crop_images_for_classification.py --root_dir  '/home/dorian/Code/turtles/turtle_datasets/031216amnorth/obj_train_data'
python crop_images_for_classification.py --root_dir  '/home/dorian/Code/turtles/turtle_datasets/031216AMsouth/obj_train_data'
python crop_images_for_classification.py --root_dir  '/home/dorian/Code/turtles/turtle_datasets/081220-0310AMsouth-ME2D/obj_train_data'


