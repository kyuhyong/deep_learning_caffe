'''
Title           :capture_label.py
Description     :
Author          :Kyuhyong
Date Created    :20180127
Date Modified   :
version         :0.1
usage           :
python_version  :2.7.11
'''

import cv2
import argparse
import os, errno
import glob

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

labelNum = 4
labelNames = ['moveCenter','moveLeft', 'moveRight', 'spinLeft']
captureKey = ['s','a','d','w']


if len(labelNames) != labelNum or len(labelNames) != labelNum :
    print "Number of labels mis-match to label names"
    exit(-1)

labelCnt = [0 for x in range(labelNum)]
fnames = []
# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument( "-f", "--folder", required=True,
	help="path to folder images to be stored")
parser.add_argument( "-s", "--size", required=False,
	help="Size of image to be captured")

args = parser.parse_args()
capturePath = os.path.dirname(args.folder+"/")

# Check if input folder is exist and count file names with labels
if not os.path.exists(capturePath):
    print "Making directory :" + capturePath
    os.makedirs(capturePath)
else :
    fnames = [img for img in glob.glob(capturePath+"/*jpg")] 
    print "Path to "+capturePath+" exist!"
    maxNum = [0 for x in range(labelNum)]
    for fname in fnames:
        for idx, label in enumerate(labelNames):
            if label in fname:
                # Search for dot(.) separated number in file name
                foundNum = int((fname.split(".",1)[1]).rstrip(".jpg"))
                if maxNum[idx] < foundNum:
                    maxNum[idx] = foundNum;
                labelCnt[idx] += 1
    for idx, num in enumerate(labelCnt):
        if num < maxNum[idx]:
            labelCnt[idx] = maxNum[idx] + 1
        print labelNames[idx]+": {:0d}".format(labelCnt[idx])

if args.size is not None:
    if not (args.size).isdigit():
        print "Input image size is not(wrong) number. Exit"
        exit(-1)
    IMAGE_WIDTH = int(args.size)
    print "Image size is set to : {:0d}".format(IMAGE_WIDTH)
else:
    print "Image size is set deafult to : {:0d}".format(IMAGE_WIDTH)
#Start capture video from Camera
cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1180, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
#cap = cv2.VideoCapture(-1)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Flip frame
    frame = cv2.flip(frame, -1)
    frame = cv2.flip(frame, 1)
    # Get frame size
    img_H, img_W = frame.shape[:2]
    resize_R = float(img_H)/img_W
    #print resize_R
    frame_roi = cv2.resize(frame, None, fx=resize_R, fy=1, interpolation=cv2.INTER_AREA)
    #Image Resizing
    frameCapture = cv2.resize(frame_roi, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
    # Display the resulting frame
    cv2.imshow('frame',frame_roi)
    # Check key capture
    waitkey = cv2.waitKey(1) & 0xFF
    for idx, key in enumerate(captureKey):
        if waitkey == ord(key):
            fname = capturePath+'/'+labelNames[idx]+'.{:0d}.jpg'.format(labelCnt[idx])
            cv2.imwrite(fname, frameCapture)
            labelCnt[idx] += 1
            print "Image saved: "+fname
    if waitkey == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
