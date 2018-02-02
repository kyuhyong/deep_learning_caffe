'''
Title           :testVideo.py
Description     :
Author          :Kyuhyong
Date Created    :20180127
Date Modified   :
version         :0.1
usage           :
python_version  :2.7.11
'''

import cv2

#Start capture video from Camera
cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1180, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Flip frame
    frame = cv2.flip(frame, -1)
    frame = cv2.flip(frame, 1)
    # Get frame size
    img_H, img_W = frame.shape[:2]
    #print "Image size = {:0d}, {:0d}".format(img_W, img_H)
    resize_R = float(img_H)/img_W
    #print resize_R
    #frame_roi = frame[0:img_W, img_H/2 - img_W : img_H/2 + img_W ]
    frame_roi = cv2.resize(frame, None, fx=resize_R, fy=1, interpolation=cv2.INTER_AREA)
    # Display the resulting frame
    cv2.imshow('frame',frame_roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
