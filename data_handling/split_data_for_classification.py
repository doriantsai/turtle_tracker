#! /usr/bin/env/python3

# assuming images are cropped from crop_images_for_classification.py
# in similar folder structure
# gather 'round and autosplit into the format

# shortcut: copy + paste all images into one big folder
# we can get away with this because image classification dataset/image sizes are small


# output given by:
# https://docs.ultralytics.com/datasets/classify/

import os
import glob
import shutil
from sklearn.model_selection import train_test_split
import code

# ################ helper functions #####################

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
       
       
def sort_image_list_to_dir(img_list, data_split, data_dirs, dir_name):
    [train_percent, test_percent, val_percent] = data_split
    [ train_dir, test_dir, val_dir] = data_dirs
    # Split data using scikit-learn's train_test_split
    train_list, temp_list = train_test_split(img_list, test_size=(1 - train_percent))
    test_list, val_list = train_test_split(temp_list, test_size = test_percent / (test_percent + val_percent))
    
    for source_list, destination in [(train_list, train_dir), (test_list, test_dir), (val_list, val_dir)]:
        
        os.makedirs(os.path.join(destination, dir_name), exist_ok=True)
        clear_files_in_folder(os.path.join(destination, dir_name))
        
        for img_file in source_list:
            source_path = img_file
            destination_path = os.path.join(destination, dir_name, os.path.basename(img_file))
            shutil.copy(source_path, destination_path)
            
    return train_list, test_list, val_list
      
# ################ parameters #####################

            
root_dir = '/home/dorian/Code/turtles/turtle_datasets/classification'

painted_dir = os.path.join(root_dir, 'all_images', 'painted_turtles')
unpainted_dir = os.path.join(root_dir, 'all_images', 'unpainted_turtles')

# painted/unpainted dictionary:
labels = ['unpainted', 'painted']

# output directories
train_dir = os.path.join(root_dir, 'train')
test_dir = os.path.join(root_dir, 'test')
val_dir = os.path.join(root_dir, 'val')
data_dirs = [train_dir, test_dir, val_dir]

# percentage split
train_percent = 0.7
test_percent = 0.15
val_percent = 0.15
data_split = [train_percent, test_percent, val_percent]

#################

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# clear folders after each run just to make sure we don't accummulate files over successive runs
clear_files_in_folder(train_dir)
clear_files_in_folder(test_dir)
clear_files_in_folder(val_dir)

# grab list of all images files (YOLO)
painted_list = sorted(glob.glob(os.path.join(painted_dir, '*.jpg')))
unpainted_list = sorted(glob.glob(os.path.join(unpainted_dir, '*.jpg')))
print(f'number of painted: {len(painted_list)}')
print(f'number of unpainted: {len(unpainted_list)}')

# TODO make this  a function - do the same for each list
# first, for the painted:

up_train_list, up_test_list, up_val_list = sort_image_list_to_dir(unpainted_list, data_split, data_dirs, labels[0])
pt_train_list, pt_test_list, pt_val_list = sort_image_list_to_dir(painted_list, data_split, data_dirs, labels[1])


print(f'Training:')
print(f'Painted: {len(pt_train_list)}')
print(f'Unpainted: {len(up_train_list)}')

print(f'Validation:')
print(f'Painted: {len(pt_val_list)}')
print(f'Unpainted: {len(up_val_list)}')

print(f'Testing:')
print(f'Painted: {len(pt_test_list)}')
print(f'Unpainted: {len(up_test_list)}')

print('done')
code.interact(local=dict(globals(),**locals()))



