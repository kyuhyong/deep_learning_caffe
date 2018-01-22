'''
Title           :make_predictions_1.py
Description     :This script makes predictions using the 1st trained model and generates a submission file.
Author          :Adil Moujahid, Kyuhyong You
Date Created    :20160623
Date Modified   :20180122
version         :0.3
usage           :python make_predictions_1.py
python_version  :2.7.11
Input arguments :[1] Path to input which contains mean binary image 
                 [2] Path to trained caffe model
                 [3] Path to test folder where unseen images are located (target files)
'''

import os
import sys
import shutil
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
inputPath = sys.argv[1]
sys.path.insert(1, inputPath)
meanBinary = inputPath + 'mean.binaryproto'
if not os.path.exists(meanBinary):
    print "Mean binary file not exist."
    sys.exit(0)
with open(meanBinary) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))
modelPath = sys.argv[2]
sys.path.insert(1, modelPath)
caffeProto = modelPath + 'caffenet_deploy_1.prototxt'
if not os.path.exists(caffeProto):
   print "Prototxt file not exist"
   sys.exit(0)
caffeModel = modelPath + 'caffe_model_1_iter_10000.caffemodel'
if not os.path.exists(caffeModel):
   print "Caffe model not exists"
   sys.exit(0)
#Read model architecture and trained model's weights
net = caffe.Net(caffeProto,
                caffeModel,
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
testPath = sys.argv[3]
sys.path.insert(1, testPath)
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob(testPath+"/*jpg")]

#Making predictions
predPath = testPath + 'predict'
if os.path.exists(predPath):
    os.system('rm -rf  ' + predPath)
os.makedirs(predPath)
dir1path = predPath + '/c1'
dir2path = predPath + '/c2'
os.makedirs(dir1path)
os.makedirs(dir2path)
test_ids = []
preds = []
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    preds = preds + [pred_probas.argmax()]
    argmax = pred_probas.argmax()
    if argmax == 1:
        shutil.copy2(img_path, dir1path)
    else:
        shutil.copy2(img_path, dir2path)
    print img_path
    print pred_probas.argmax()
    print '-------'

'''
Making submission file
'''
with open(modelPath +"submission_model_1.csv","w") as f:
    f.write("id,label\n")
    for i in range(len(test_ids)):
        f.write(str(test_ids[i])+","+str(preds[i])+"\n")
f.close()
