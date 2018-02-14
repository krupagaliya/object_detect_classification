# object_detect_classification

Object Detection and Classification with SSD and MobileNet model

Detect the object and classify that objects in following category

["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","House"]

## Requirements
Python 3.5
Numpy 1.14.0
OpenCV 3.4.0


## Run the code
python deep_learning_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --image image_path
