import numpy as np
import argparse
import cv2
import sys
import xml.etree.ElementTree as ET

# constructing the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")

ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")


ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","House"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


print("[INFO] loading serialized model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

for i in np.arange(0, detections.shape[2]):
	
	
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# extract the index of the class label from the `detections`,
		# then compute coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		print("[INFO] {}".format(label))
		cv2.rectangle(image, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(image, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

#Display image
cv2.imshow("Output", image)
#saving .png file to working directory
cv2.imwrite('output_img.png', image)

#print some extra information too
print("File name ",sys.argv[6])
print("xmin",startX)
print("ymin",startY)
print("xmax",endX)
print("ymax",endY)
print("Width", w)
print("Height",h)
File_name= sys.argv[6]

#XML file generation
root = ET.Element("annotation")

ET.SubElement(root, "filename", name="filename").text = File_name

size = ET.SubElement(root, "size", name="size")
ET.SubElement(size, "width", name="width").text = w
ET.SubElement(size, "height", name="height").text = h

object1 = ET.SubElement(root, "object", name="object")
box = ET.SubElement(object1, "bndbox", name="bndbox")
ET.SubElement(box, "xmin", name="xmin").text = startX
ET.SubElement(box, "ymin", name="ymin").text = startY
ET.SubElement(box, "xmax", name="xmax").text = endX
ET.SubElement(box, "ymax", name="ymax").text = endY

tree = ET.ElementTree(root)
print(tree)
tree.write("submit.xml")

#or press q to exit
cv2.waitKey(0)
