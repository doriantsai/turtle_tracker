## Detection on Yolov8
For detecting turtles in an image (or set of images or video) with the possibility to show detections. 
Also can be used to validate training results as the detector can also show annotated data.

To run the detector: 
- edit Dectect.py (in the detector folder) so that the weights file, yolo_dir, save_dir and image_dir are pointing to the data.
- edit Dectect.py so that the TurtleDetector.run has the correct specifications (ie, save_imgs and show_imgs set to true if required)
- run Detect.py with the following command:

      python detector/Detect.py 

You can also use the ultralitics yolov5 scripts which are compatible
- https://github.com/doriantsai/yolov5_turtles/tree/master
