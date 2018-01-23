'''
Title           :rt_classification.py
Description     :This script runs realtime image classification with caffe models
Author          :Kyuhyong
Date Created    :20180123
Date Modified   :20180123
version         :0.1
usage           :
python_version  :2.7.11
'''
import numpy as np
import sys
import argparse
import glob
import cv2
import caffe
from caffe.proto import caffe_pb2

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

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
mean_blob = caffe_pb2.BlobProto()
with open(args.mean) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))
caffeProto = args.prototxt
caffeModel = args.model
#Read model architecture and trained model's weights
net = caffe.Net(caffeProto,
                caffeModel,
                caffe.TEST)
cap = cv2.VideoCapture(0)
#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = transform_img(frame, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']
    argmax = pred_probas.argmax()
    if argmax == 1:
        print 'Looks like DOG with Probability of {:0.2f}'.format(pred_probas[0][1])
    else:
        print 'Looks like CAT with Probability of {:0.2f}'.format(pred_probas[0][0])
    print 
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
