#!/usr/bin/env python
# license removed for brevity
import rospy
from my_classifier.msg import Num

command = Num()

import numpy as np
import sys
import argparse
import glob
import cv2
import caffe
from caffe.proto import caffe_pb2
# Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

# Write some Text
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,600)
fontScale              = 0.8
fontColor              = (255,0,0)
lineType               = 2

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def talker():
    command.linear = 0.5
    command.rotation = 0.1
    command.myid = 11
    command.prob = 0.9
    pub = rospy.Publisher('my_classifier', Num, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(15) # 10hz
    while not rospy.is_shutdown():
        #hello_str = "hello world %s" % rospy.get_time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Flip frame to adjust horizon
        frame = cv2.flip(frame, -1)
        frame = cv2.flip(frame, 1)
        img_H, img_W = frame.shape[:2]
        # Crop image to rectangular shape --> Old method
        #frame_roi = frame[0:img_W, img_H/2 - img_W : img_H/2 + img_W ]
        # Resize image to rectangular shape --> New method
        frame_roi = cv2.resize(frame, None, fx=float(img_H)/img_W, fy=1.0, interpolation=cv2.INTER_AREA)
        img = transform_img(frame_roi, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        pred_probas = out['prob']
        argmax = pred_probas.argmax()
        if argmax == 1:
            probability = pred_probas[0][1]
            scr_msg = 'Move Left P:  {:0.2f}'.format(probability)
            command.linear = 0.6
            command.rotation = 0.6
            command.prob = probability
        elif argmax ==2:
            probability = pred_probas[0][2]
            scr_msg = 'Move Right P: {:0.2f}'.format(probability)
            command.linear = 0.6
            command.rotation = -0.6
            command.prob = probability
        elif argmax ==3:
            probability = pred_probas[0][3]
            scr_msg = 'Spin Left P: {:0.2f}'.format(probability)
            command.linear = 0.0
            command.rotation = 0.5
            command.prob = probability
        else:
            probability = pred_probas[0][0]
            scr_msg = 'Move Center P:  {:0.2f}'.format(probability)
            command.linear = 0.8
            command.rotation = 0.0
            command.prob = probability
        # Set Display message
        roi_H, roi_W = frame_roi.shape[:2]
        cv2.rectangle(frame_roi, (0,int(roi_H*0.9)), (roi_W, roi_H), (255,255,255), -1)
        bottomLeftCornerOfText = (int(roi_W * 0.05), int(roi_H * 0.95))
        cv2.putText(frame_roi, scr_msg, 
	        bottomLeftCornerOfText, 
	        font, 
	        fontScale,
	        fontColor,
	        lineType)
        # Display the resulting frame
        cv2.imshow('frame',frame_roi)
        #command.linear += 0.1
        #if(command.linear>10.0):
        #    command.linear=0.0
        if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
        rospy.loginfo(command)
        pub.publish(command)
        rate.sleep()

caffe.set_mode_gpu() 

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument( "-b", "--mean", required=True,
	help="path to mean binary image")
parser.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
parser.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
args = parser.parse_args()
#Read mean image
print "Reading mean binary..."
mean_blob = caffe_pb2.BlobProto()
with open(args.mean) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

#Read model architecture and trained model's weights
print "Making model architecture..."
net = caffe.Net(args.prototxt,
	        args.model,
	        caffe.TEST)
#Define image transformers
print "Defining image transformers..."
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))
#Start capture video from Camera
cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1180, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

