# Classifier Plans

# Train Classifier
- get cropped images of the turtles (painted/unpainted) from detector (+- some bounding box)
- create balanced dataset of cropped images for painted/unpainted turtles
- train classifier: for basic approach, can use ConvNet (see https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), or for SOTA, can use NFNets (https://github.com/benjs/nfnets_pytorch)
- during operation, can give it single image for classification, but better yet may be giving sequence of images (i.e. track), do classification on each image and then vote for classification over the sequence
- output: image/sequence is painted/unpainted turtle


- follow the data format for Yolov5:

        dataset/
        ├── train/
        ├──── class1/
        ├──── class2/
        ├──── class3/
        ├──── ...
        ├── val/
        ├──── class1/
        ├──── class2/
        ├──── class3/
        ├──── ...

- download yolov5 classification checkpoints (specifically the classification model) from https://github.com/ultralytics/yolov5#pretrained-checkpoints, currently using YOLOv5x-cls, but hope to go for YOLOv5n-cls for faster runtimes

- using Yolov5, the training command (note: change the relevant files/options):

       python classify/train.py --data job10_classification --model weights/yolov5x-cls.pt --epochs 20 --img 224 --cache 



# Updates for YOLOv8

- Classifierv8.py is updated to use Yolov8

- get classification images (cropped from original annotations) via the following scripts in `data_handling`:

        crop_images_for_classification.py

- the script `bash_crop_for_class.sh` runs `crop_images_for_classification.py` several times for different folders

- copy/paste all the painted/unpainted turtle images into a single folder, as per the data format above (`dataset/painted`, `dataset/unpainted`), then run the script below to split the data into train/val/test (whilst setting the appropriate folder names in the script):

        split_data_for_classification.py

- in the Yolov8 install folder (see main Readme.md for details on how to install), run classify_turtles.py