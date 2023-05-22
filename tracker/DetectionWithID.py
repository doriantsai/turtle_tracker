
# TODO should probably inherit from Detector class definition and just add id property
class DetectionWithID():
    
    def __init__(self, class_label, box, confidence, id, image_name):
        self.class_label = class_label
        self.box = box
        self.detection_confidence = confidence
        self.id = id
        self.image_name = image_name
        