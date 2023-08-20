# turtle_tracker
To automatically count sea turtles, both painted and unpainted turtles from drone imagery, we have developed a turtle detector, classifier and tracker.
We have useed yolov5 for classification and yolov8 for detection and tracking.

## Installation & Dependencies
- The following code was run in Ubuntu 20.04 LTS using Python 3
- Install conda https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html, although I recommend miniforge/mambaforge https://github.com/conda-forge/miniforge, which is a much more minimal and faster installer for conda packages. These installers will create a ''base'' environment that contains the package managers for conda.
- `conda install mamba -c conda-forge`
- `git clone git@github.com:doriantsai/turtle_tracker.git`
- `cd` to the turtle_tracker folder
- Create conda environment automatically by running the script "make_conda_turtles.sh" file
    - make .sh file executable via `chmod +x make_conda_turtles.sh`
    - run .sh file via `./make_conda_turtles.sh`
    - Note: you can read the dependencies in the `turtles.yml` file
- this should automatically make a virtual environment called "turtles"
- to activate new environment: `conda activate turtles`
- For completness, it is also may be helpful to
    - `pip install -e.` and `pip install ultralytics -U`
- to deactivate the new environment: `conda deactivate`

## WandB
- Weights & Biases is used to track and visualise the model training progress
- Setup an account https://docs.wandb.ai/quickstart

## Counting turtles from a video
To count the painted and non-painted turtles from a video, do the following:
- edit the pipeline_config.yaml file (in the tracker folder) to point to the data.
- run TurtleTrackingPipeline.py with the following command:

        python tracker/TurtleTrackingPipeline.py

## Training on Yolov8
This repo has none of the training scripts to train yolo. However, ultalitics yolov8 https://github.com/ultralytics/ultralytics works very well. You will need to:
- activate turtles environment
- git clone yolov8 from ultralytics github
- download the relevant pre-trained weights file (yolov8l.pt)
- setup training/testing/validation data splits
- make sure classes are set (e.g. all turtles vs painted/non-painted)
- setup relevant .txt files for training/testing across multiple folders
- setup data.yml for relevant dataset(s)
- run train.py with the following command (note: change the relevant files/options): 

        python train.py --data turtles_job10_041219-0-1000.yml --weights weights/yolov8l.pt --img 1280 --batch 10 --epochs 10 --cache ram

- Notes on training on the turtle images. The variation in the quality of blues and greens are very high in ocean-related imagery. Therefore, it is highly recommended to train with colour jitter (variation in saturation, hue, etc). Also, random resize recrop for different drone heights. By default, YOLOv8 does not have this enabled. Follow the following steps to enable this functionality:

        - remove the ultralytics install from conda (conda uninstall ultralytics) if one install already exists (or pip uninstall ultralytics)
        - clone yolov8 repo
        - cd to the yolov8 repo, inside the requirements.txt, uncomment the `albumentations` option underneath `Extras`
        - cd to the yolov8 repo, `pip install -e .` to install the yolov8 repo locally
        - find the augment.py file in (yolov8 folder)/ultralytics/data/augment.py
        - change the 
        - NOTE: in this file, we can also change the size of the expected box candidates to much smaller (turtles = small in the drone imagery) by adjusting the `box_candidates` area threshold (area_thr) from 0.1 to 0.05 or similar.
        - in the class Albumentations, add 

                def __init__(self, p=1.0, size=640):

        - as well as T for the transforms (replacing the current/old `T`):

                T = [
                        A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.5),
                        A.Blur(p=0.01),
                        A.MedianBlur(p=0.01),
                        A.ToGray(p=0.01),
                        A.CLAHE(p=0.01),
                        A.ColorJitter(brightness=0.5,
                                contrast=0.4,
                                saturation=0.4,
                                hue=0.4,
                                p=0.6),
                        A.RandomBrightnessContrast(p=0.0),
                        A.RandomGamma(p=0.0),
                        A.ImageCompression(quality_lower=75, p=0.0)]

        - save this, and then when you initiate traiing, you should be able to see the transform/albumentations/jitter settings at the start of the output in the terminal
        

## Detection on Yolov8
- edit Dectect.py (in the detector folder) so that the weights file, yolo_dir, save_dir and image_dir are pointing to the data.
- edit Dectect.py so that the TurtleDetector.run has the correct specifications (ie, save_imgs and show_imgs set to true if required)  
- run Detect.py with the following command:

        python detector/Detect.py 

## Tracking on Yolov8
- edit Tracker.py (in the tracker folder) so that the weights file, yolo_dir, save_dir and video_dir are pointing to the data
- run Tracker.py with the following command:

        python tracker/Tracker.py

## Classification on Yolov5
- edit Classify.py (in the classifier folder) so that the weights file, yolo_dir, img_dir are pointing to the data.
- run Detect.py with the following command:

        python classifier/Classify.py
  
