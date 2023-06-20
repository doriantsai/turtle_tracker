# Tracking
For tracking you can use strongsort or https://github.com/doriantsai/yolov8_tracking_turtles
Also is this repo we've created a file that tracks over a video `TurtleTrackerPipeline.py`

## Counting turtles from a video
To count the painted and non-painted turtles from a video, do the following:
- edit the pipeline_config.yaml file (in the tracker folder) to point to the data.
- run TurtleTrackingPipeline.py with the following command:

      python tracker/TurtleTrackingPipeline.py

## Other functionality
This folder also contains classes to store turtle data.
- DetectionWithID - stores a class label, array of location (x1,y1,x2,y2 normalised), detection confidence, track id and the image name the turtle is in
- ImageTrack - stores the track of a specific turtle, with a list of DetectionWithID over a series of images as well as adding in classification and classification condidence data and an overall classification
- ImageWithDetection - stores a txt_file, image name, DetectionWithID data and the image height and width.
- ImageWithDetectionTrack = stores the track of a specific turtle with a list of ImageWithDetection over a series of images as well as adding in classification and classification condidence data and an overall classification
