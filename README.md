# deep_learning_caffe

All the codes and scripts are originated from **Adil Moujahid's** post in http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/ and his github page found in https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial and slightly modified for my own caffe tutorial.

## Repository intro
The repository is consist of 3 folder as follows
 - caffe model: Contains definitions of train and solver prototypes for use in caffe
 - input: Empty folder where all the dataset used for training, validtion and classification of unseen images.
 - pycode: Includes python codes for making LMDB files, plot learning curve and make predictions.

## Requirements
In order to run the scripts and train with caffe, your machine needs to be set up as follows.
 - OS: Ubuntu 16.04
 - opencv: At least v3.0.0 or later is installed
 - Caffe installed either GPU or CPU support: To check this. Run python and try **import caffe** and see if no errors found.
 - Python modules: pip, numpy, lmdb, graphviz, pandas

If you need detailed description on how to install Caffe with GPU suport, see http://kyubot.tistory.com/93?category=617700

## Where is the tutorials
To learn how deep learning is work for image classifications, please see tutorial in http://kyubot.tistory.com/96?category=617700 
To learn how to train deep learning model and use for pediction, please see my tutorial in http://kyubot.tistory.com/97?category=617700

## How to run
All the commands that need to entered in terminal are in https://github.com/kyuhyong/deep_learning_caffe/blob/master/terminal_command

For cat and dog classification, you will need to download dataset as **train.zip** and **test1.zip** which can be found in https://www.kaggle.com/c/dogs-vs-cats/data
Copy the above files into input folder and unzip as follows
```{r, engine='sh'}
cd ~/deep_learning_caffe 
cd input 
unzip ~/deeplearning-cats-dogs-tutorial/input/train.zip 
unzip ~/deeplearning-cats-dogs-tutorial/input/test1.zip 
rm ~/deeplearning-cats-dogs-tutorial/input/*.zip
```
Go to pycode folder and try run create_train_lmdb.py
```{r, engine='sh'}
cd ../pycode
python create_train_lmdb.py ~/deep_learning_caffe/input/
```
Generate mean image by running an app in caffe tools
```{r, engine='sh'}
cd ~/caffe/build/tools
./compute_image_mean -backend=lmdb ~/deep_learning_caffe/input/train_lmdb/ ~/deep_learning_caffe/input/mean.binaryproto
```
Modify **path to mean_file, source** in model defintion files under **/caffe_models/caffe_model_1** per the name of your home directory.

To print model architecture, run a python script contained in caffe as follows
```{r, engine='sh'}
python ~/caffe/python/draw_net.py ~/deep_learning_caffe/caffe_models/caffe_model_1/caffenet_train_val_1.prototxt ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1.png
```
To train the model run caffe tools as follows,
```
cd ~/caffe/build/tools
./caffe train --solver ~/deep_learning_caffe/caffe_models/caffe_model_1/solver_1.prototxt 2>&1 | tee ~/deep_learning_caffe/caffe_models/caffe_model_1/model_1_train.log
```
To print learning curve to see how well our model trained, run python script in project's pycode folder
```
python ~/deep_learning_caffe/pycode/plot_learning_curve.py ~/deep_learning_caffe/caffe_models/caffe_model_1/model_1_train.log ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1_learning_curve.png
```
![alt text] (http://cfile22.uf.tistory.com/image/99705D4C5A60DB7805059D "Learning curve")

To predict image classifiction from test images, run python script as follows
```sh
python make_predictions_1.py ~/deep_learning_caffe/input/ ~/deep_learning_caffe/caffe_models/caffe_model_1/ ~/deep_learning_caffe/input/test1/
```
This will create a folder named predict under test folder and put classified images in a seperate folders. 

For any request or feedback, send an email to kyuhyong [at] gmail [dot] com
Thank you.

