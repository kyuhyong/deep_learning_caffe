
# coding: utf-8
# File name: create_train_lmdb.py
# Author: Kyuhyong
# Date created: 1/16/2018
# Last modified: 1/16/2018
# Python Version: 2.7
# Description:
# Creates LMDB file from following train data structure under user input directory
# root of input/train/cat.###.jpg      label: 0
#                    /dog.###.jpg      label: 1
# LMDB file will be generated under user input folder
# This code is modified from the code originally created by Adil Moujahid

import os
import sys
import glob
import random
import numpy as np
import cv2
from caffe.proto import caffe_pb2
import lmdb

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

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

pathToInput = sys.argv[1]
sys.path.insert(1, pathToInput)
train_lmdb = pathToInput+'train_lmdb'
validation_lmdb = pathToInput+'validation_lmdb'
pathToTrain = pathToInput+'train'
if not os.path.isdir(pathToTrain):
	print "No train folder exist. Exit."
	sys.exit(0)
print "Generate train lmdb to {}".format(train_lmdb)
os.system('rm -rf  ' + train_lmdb)
in_db = lmdb.open(train_lmdb, map_size=int(1e12))
train_data = [img for img in glob.glob(pathToTrain+"/*jpg")]
dataNum = len(train_data)
#Shuffle train_data
random.shuffle(train_data)
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
	if in_idx % 6 == 0:		# 5/6 of train data used for train
		continue
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
	if 'moveCenter' in img_path:
	    label = 0
        elif 'moveLeft' in img_path:
            label = 1
        elif 'moveRight' in img_path:
            label = 2
	else:
	    label = 3
	datum = make_datum(img, label)
	in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
	print '{:d} of {:d}|'.format(in_idx,dataNum) + os.path.basename(os.path.dirname(img_path)) + "/"+ os.path.basename(img_path) + ' as {:d}'.format(label),"          \r",
in_db.close()
print "\nTraining LMDB is created in "+train_lmdb

print "Generate validation lmdb to {}".format(validation_lmdb)
os.system('rm -rf  ' + validation_lmdb)
in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
	if in_idx % 6 != 0:		# only 1/6 of train data used for validation
		continue
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
	if 'moveCenter' in img_path:
	    label = 0
        elif 'moveLeft' in img_path:
            label = 1
        elif 'moveRight' in img_path:
            label = 2
	else:
	    label = 3
	datum = make_datum(img, label)
	in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
	print '{:d} of {:d} @'.format(in_idx, dataNum) + os.path.basename(os.path.dirname(img_path)) + "/"+ os.path.basename(img_path) + ' as {:d}'.format(label),"             \r",
in_db.close()
print "\nValidation LMDB is created in " + validation_lmdb

