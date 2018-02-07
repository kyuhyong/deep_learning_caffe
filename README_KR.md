# deep_learning_caffe
이 코드는 Caffe를 활용한 딥러닝 이미지 Classification 튜토리얼 http://kyubot.tistory.com/97?category=617700 을 응용하여 실내 자율주행을 위한 코드를 포함하고 있습니다.
일부 코드는  **Adil Moujahid's** 이 본인의 블로그에 포스팅한 http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/ 의 자료를 바탕으로 작성되었으며 이곳 https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial 일부가 수정 및 추가되었음을 밝힙니다.


## Repository 소개
이 저장소는 4개의 폴더로 구성되어있습니다.
 - caffe model: caffe 에서 사용되는 신경망에 대한 정의 및 solver 프로토타입을 포함합니다.
 - input: 트레이닝 및 평가용 데이터 셋 그리고 Prediction을 위한 테스트 이미지가 저장될 빈 폴더입니다.
 - pycode: LMDB 생성, Learning curve를 그리기 위한 모든 파이썬 코드에 더하여 카메라 이미지를 라벨링 하여 캡춰하는 기능이 추가되었습니다.
  - ros_catkinws: ROS가 설치되어 있는 환경에서 인공지능을 활용하여 실제 로봇을 구동하기 위한 코드들을 포함하고 있습니다.

## 요구사항
이곳의 파이썬 스크립트, Caffe 모델을 사용하기 위해서는 기본적으로 다음 사항을 만족해야 합니다.
 - 운영체제 : Ubuntu 16.04 (다른 버전 혹은 운영체제에서는 테스트하지 않았습니다.)
 - opencv: 최소 버전 v3.0.0 이 설치됨
 - CPU혹은 GPU가 지원되는 Caffe 버전 설치: 파이썬에서  **import caffe** 를 입력하여 아무 오류가 없는지 확인합니다. GPU가 지원되지 않는 경우 스크립트 일부를 수정해야할 수 있습니다.
 - 파이썬 모듈: pip, numpy, lmdb, graphviz, pandas
 - **ROS Kinetic**: ROS Kinetic버전(테스트됨)이 설치되어 있어야합니다. 자세한 설치 방법은 http://wiki.ros.org/kinetic/Installation 를 참조하기 바랍니다. 

GPU를 지원하는 Caffe를 설치하는 보다 자세한 설명은 제 블로그의 튜토리얼  http://kyubot.tistory.com/93?category=617700 을 참조하세요.

## 참조 튜토리얼
이미지 Classification이 어떻게 동작하는지 알기 위해서는 다음을 참조하세요. http://kyubot.tistory.com/96?category=617700 
딥러닝 모델을 훈련시키고 예측하는 방법은 다음을 참조하세요. http://kyubot.tistory.com/97?category=617700

## 실행 방법
가장 먼저 git clone 명령어로 git 저장소를 여러분의 홈 디렉토리에 다운로드 합니다.
```
git clone https://github.com/kyuhyong/deep_learning_caffe.git
```
이렇게 하면 폴더에 deep_learning_caffe 라는 폴더가 생성될 것입니다.

여기에 포함된 모든 실행 명령어는 다음 페이지 https://github.com/kyuhyong/deep_learning_caffe/blob/master/terminal_command 혹은 메인 폴더의 terminal_command 에 있습니다. 
저장소 폴더로 이동하여 *checkout* 명령어로 저장소 branch를 **agvnet** 으로 변경합니다.
```
cd ~/deep_learning_caffe
git checkout agvnet

Branch agvnet set up to track remote branch agvnet from origin.
Switched to a new branch 'agvnet'
```
이제 기존의 개/고양이 식별을 위한 데이터셋을 다운로드 하는 대신 딥러닝을 훈련하기 위한 새로운 데이터셋을 만들어야 합니다.
카메라로부터 들어온 복도 영상에서 좌/우/가운데/벽을 인식하기 위해 pycode의 **capture_label.py**를 사용하여 라벨링된 이미지들을 저장합니다.
```
cd ../pycode
python capture_label.py -f ../input/train
```
이렇게 하면 input 폴더 안에 train 이라는 폴더가 생성되고 각 라벨 별로 넘버링된 .jpg 파일이 생성된 것을 확인할 수 있습니다.

이제 만들어진 훈련 데이터로부터 LMDB 파일을 생성하기 위해 create_train_lmdb.py 을 실행합니다.
```
python create_train_lmdb.py ~/deep_learning_caffe/input/
```
caffe tool 폴더에서 프로그램을 실행하여 mean image를 만듭니다.
```
cd ~/caffe/build/tools
./compute_image_mean -backend=lmdb ~/deep_learning_caffe/input/train_lmdb/ ~/deep_learning_caffe/input/mean.binaryproto
```
**/caffe_models/caffe_model_1** 안에 포함된 모델 정의 파일을 에디터에서 열고  **path to mean_file, source** 경로 등을 여러분의 홈 디렉토리 경로에 맞게 수정합니다.

모델 아키텍쳐를 그림으로 나타내기 위해서는 아래 명령어를 입력합니다. 첫번째 파라미터는 해당 모델의 정의파일이고 두번째는 생성된 그림이 저장될 이름입니다.
```
python ~/caffe/python/draw_net.py ~/deep_learning_caffe/caffe_models/caffe_model_1/caffenet_train_val_1.prototxt ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1.png
```
정의된 모델을 훈련시키기 위해서는 caffe tool 폴더에서 다음과 같이 실행합니다. 
```
cd ~/caffe/build/tools
./caffe train --solver ~/deep_learning_caffe/caffe_models/caffe_model_1/solver_1.prototxt 2>&1 | tee ~/deep_learning_caffe/caffe_models/caffe_model_1/model_1_train.log
```
실행 과정에서 생성되는 로그는 **model_1_train.log** 파일에 저장되게 됩니다.

위에서 생성된 로그를 바탕으로 모델이 얼마나 잘 훈련되었는지 확인하기 위해서는 pycode 폴더에서 아래와 같이 파이썬 스크립트를 실행하여 러닝커브를 그려봅니다.
```
python ~/deep_learning_caffe/pycode/plot_learning_curve.py ~/deep_learning_caffe/caffe_models/caffe_model_1/model_1_train.log ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1_learning_curve.png
```
![alt text] (http://cfile22.uf.tistory.com/original/99705D4C5A60DB7805059D#jpg "Learning curve")

훈련된 모델을 가지고 라벨링 되지 않은 테스트 이미지를 식별하기 위해서는 pycode 폴더에서 아래와 같이 실행합니다.
여기서 caffe_model_1_iter_10000.caffemodel 은 **epoch** 를 10000으로 설정했을때 생성되는 파일로서 다른 값을 입력한 경우 이름은 달라지게 됩니다.
```sh
$ python make_predictions_1.py --mean ~/deep_learning_caffe/input/mean.binaryproto --prototxt ~/deep_learning_caffe/caffe_models/caffe_model_1/caffenet_deploy_1.prototxt --model ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1_iter_10000.caffemodel --test ~/deep_learning_caffe/input/test1/
```
이제 식별된 이미지들이 각각의 폴더에 저장될 것입니다.

만약 컴퓨터에 USB카메라등이 장착되어 있다면 아래와 같이 rt_classification.py를 실행하여 실시간으로 들어오는 영상을 식별하는 것을 확인할 수 있습니다.
```sh
$ python rt_classification.py --mean ~/deep_learning_caffe/input/mean.binaryproto --prototxt ~/deep_learning_caffe/caffe_models/caffe_model_1/caffenet_deploy_1.prototxt  --model ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1_iter_10000.caffemodel
```
질문이나 의견 혹은 bug 리포트 등은 제 이메일 kyuhyong [at] gmail [dot] com 로 보내주십시오.
감사합니다.

수요일, 07. 2월 2018 작성됨
