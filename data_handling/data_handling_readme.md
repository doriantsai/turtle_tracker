# data_handling

This folder of scripts is for organising data, changing annotations scripts, etc.

- once the data is loaded onto computer (assumed brand new download from cvat)
- run convert_classes_symlink_images.py to make turtle-only labels (from cvat, we have turtles and unpainted both)
- do this for each dataset folder
- then run cvat2trainv2.py to split the data into train/val/test folders
- TODO does not fail gracefully for empty folders
- do this for each dataset folder

- TODO this process should be wrapped up to iterate over data folders
- should auto-generate the training file
- add file locations to turltes.yaml file for training (TODO - commit these)
- run train_turtles.py in yolov8 folder (TODO - commit these)
- data should now be ready for yolov8/ultralytics training

- download yolov8 ultralytics repo and train on the data
