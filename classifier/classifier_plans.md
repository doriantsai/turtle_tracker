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

- using Yolov5, the training command:

    python classify/train.py --data /home/dorian/Data/turtle_datasets/job10_classification --model weights/yolov5x-cls.pt --epochs 20 --img 224 --cache --pretrained
