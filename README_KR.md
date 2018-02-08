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
 - 운영체제 : **Ubuntu 16.04** (다른 버전 혹은 운영체제에서는 테스트하지 않았습니다.)
 - **opencv**: 최소 버전 v3.0.0 이 설치됨
 파이썬에서 현재 설치된 Opencv 버전을 확인하려면 파이썬 환경에서 다음을 입력합니다.
 ```py
 >>> import cv2
 >>> cv2.__version__
'3.3.1'
>>>
 ```
 
 - CPU혹은 GPU가 지원되는 Caffe 버전 설치: 파이썬에서  **import caffe** 를 입력하여 아무 오류가 없는지 확인합니다. GPU가 지원되지 않는 경우 스크립트 일부를 수정해야할 수 있습니다.
 - 파이썬 모듈: pip, numpy, lmdb, graphviz, pandas
 - **ROS Kinetic**: ROS Kinetic버전(테스트됨)이 설치되어 있어야합니다. 자세한 설치 방법은 http://wiki.ros.org/kinetic/Installation 를 참조하기 바랍니다. 
 - joy node: ROS에서 조이스틱 입력을 사용하기 위해서는 joystick 모듈이 설치되어 있어야 합니다. 
```sh
 sudo apt-get install ros-kinetic-joystick-drivers 
```
설치가 완료되면 rosdep으로 설치를 진행합니다.
```sh
rosdep install joy
```


GPU를 지원하는 Caffe를 설치하는 보다 자세한 설명은 제 블로그의 튜토리얼  http://kyubot.tistory.com/93?category=617700 을 참조하세요.

## 참조 튜토리얼
이미지 Classification이 어떻게 동작하는지 알기 위해서는 다음을 참조하세요. http://kyubot.tistory.com/96?category=617700 
딥러닝 모델을 훈련시키고 예측하는 방법은 다음을 참조하세요. http://kyubot.tistory.com/97?category=617700

## Dataset 준비
가장 먼저 git clone 명령어로 git 저장소를 여러분의 홈 디렉토리에 다운로드 합니다.
```sh
git clone https://github.com/kyuhyong/deep_learning_caffe.git
```
이렇게 하면 폴더에 deep_learning_caffe 라는 폴더가 생성될 것입니다.

여기에 포함된 모든 실행 명령어는 다음 페이지 https://github.com/kyuhyong/deep_learning_caffe/blob/master/terminal_command 혹은 메인 폴더의 terminal_command 에 있습니다. 
저장소 폴더로 이동하여 *checkout* 명령어로 저장소 branch를 **agvnet** 으로 변경합니다.
```sh
cd ~/deep_learning_caffe
git checkout agvnet

Branch agvnet set up to track remote branch agvnet from origin.
Switched to a new branch 'agvnet'
```
기존의 개/고양이 식별을 위한 데이터셋을 다운로드 하는 대신 딥러닝을 훈련하기 위한 새로운 데이터셋을 만들어야 합니다.
![](https://i.imgur.com/Agwmub9.png) 
카메라로부터 들어온 복도 영상에서 좌/우/가운데/벽을 인식하기 위해 pycode의 **capture_label.py**를 사용하여 라벨링된 이미지들을 저장합니다.
```sh
cd ../pycode
python capture_label.py -f ../input/train
```
이렇게 하면 input 폴더 안에 train 이라는 폴더가 생성되고 각 라벨 별로 넘버링된 .jpg 파일이 생성된 것을 확인할 수 있습니다.

만들어진 훈련 데이터로부터 LMDB 파일을 생성하기 위해 create_train_lmdb.py 을 실행합니다.
```sh
python create_train_lmdb.py ~/deep_learning_caffe/input/
```
caffe tool 폴더에서 프로그램을 실행하여 mean image를 만듭니다.
```sh
cd ~/caffe/build/tools
./compute_image_mean -backend=lmdb ~/deep_learning_caffe/input/train_lmdb/ ~/deep_learning_caffe/input/mean.binaryproto
```
## Caffe 모델 훈련하기
이제 Caffe 모델을 훈련하기 위한 모든 준비가 되었습니다.
**/caffe_models/caffe_model_1** 안에 포함된 모델 정의 파일을 에디터에서 열고  **path to mean_file, source** 경로 등을 여러분의 홈 디렉토리 경로에 맞게 수정합니다.

모델 아키텍쳐를 그림으로 나타내기 위해서는 아래 명령어를 입력합니다. 첫번째 파라미터는 해당 모델의 정의파일이고 두번째는 생성된 그림이 저장될 이름입니다.
```sh
python ~/caffe/python/draw_net.py ~/deep_learning_caffe/caffe_models/caffe_model_1/caffenet_train_val_1.prototxt ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1.png
```
정의된 모델을 훈련시키기 위해서는 caffe tool 폴더에서 다음과 같이 실행합니다. 
```sh
cd ~/caffe/build/tools
./caffe train --solver ~/deep_learning_caffe/caffe_models/caffe_model_1/solver_1.prototxt 2>&1 | tee ~/deep_learning_caffe/caffe_models/caffe_model_1/model_1_train.log
```
실행 과정에서 생성되는 로그는 **model_1_train.log** 파일에 저장되게 됩니다.

위에서 생성된 로그를 바탕으로 모델이 얼마나 잘 훈련되었는지 확인하기 위해서는 pycode 폴더에서 아래와 같이 파이썬 스크립트를 실행하여 러닝커브를 그려봅니다.
```sh
python ~/deep_learning_caffe/pycode/plot_learning_curve.py ~/deep_learning_caffe/caffe_models/caffe_model_1/model_1_train.log ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1_learning_curve.png
```

![alt text] (https://i.imgur.com/68HVuCl.png "Learning curve")

## 훈련된 Caffe 모델로 Prediction 하기
훈련된 모델을 가지고 라벨링 되지 않은 테스트 이미지를 식별하기 위해서는 pycode 폴더에서 아래와 같이 실행합니다.
여기서 caffe_model_1_iter_10000.caffemodel 은 **epoch** 를 10000으로 설정했을때 생성되는 파일로서 다른 값을 입력한 경우 이름은 달라지게 됩니다.

```sh
python make_predictions_1.py --mean ~/deep_learning_caffe/input/mean.binaryproto --prototxt ~/deep_learning_caffe/caffe_models/caffe_model_1/caffenet_deploy_1.prototxt --model ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1_iter_10000.caffemodel --test ~/deep_learning_caffe/input/test1/
```
식별된 이미지들이 각각의 폴더에 저장될 것입니다.

만약 컴퓨터에 USB카메라등이 장착되어 있다면 아래와 같이 rt_classification.py를 실행하여 실시간으로 들어오는 영상을 식별하는 것을 확인할 수 있습니다.
```sh
$ python rt_classification.py --mean ~/deep_learning_caffe/input/mean.binaryproto --prototxt ~/deep_learning_caffe/caffe_models/caffe_model_1/caffenet_deploy_1.prototxt  --model ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1_iter_10000.caffemodel
```

## ROS 로봇 구동하기
![Logo of ROS](http://wiki.ros.org/custom/images/ros_org.png) 
ROS 는 시스템에서 여러개의 프로그램 노드들이 동작하면서 메세지를 주고받을 수 있도록 하는 하나의 프레임웍 입니다. 이 프레임웍을 통해 영상처리를 통해 나온 결과를 로봇 구동을 담당하는 노드에 쉽게 전달할 수 있습니다. 이렇게 하기 위해서는 ROS 패키징을 지원하는 Workspace를 만들고 빌드하는 과정이 필요합니다.
다음 그림은 ROS기반 모바일 로봇의 시스템 구성을 보여줍니다.

![](https://i.imgur.com/upXLnO1.png) 
---

저장소의 ros_catkinws 폴더 안에는 src  폴더가 있고 이 안에 각 노드별로 프로젝트 폴더가 존재합니다. 
 - **myjoy** : usb to serial 포트를 열어서 조이스틱 입력 혹은 my_classifier노드로부터 들어온 전진, 회전량 값에 따라서 모터 드라이버를 구동하는 노드입니다.
 - **my_classfier**: USB Camera 혹은 Jetson TX보드의 CSI카메라를 열고 실시간 영상을 Caffe모델로 식별하여 좌/우/직진 혹은 회전 명령을 생성하는 노드입니다.

각각의 폴더 안에 있는 **script** 폴더에 파이썬 코드가 들어있습니다.
###my_joy 노드
myjoy 노드의 joycon.py라는 파일을 에디터로 열면 어떻게 드라이버를 구동하는지가 나와있습니다.
먼저 joy노드로부터 Joy라는 형식의 조이스틱 메세지를 받고 my_classfier메세지로부터 Num이라는 형식의 제어 명령 메세지를 받습니다.
```py
from sensor_msgs.msg import Joy
from my_classifier.msg import Num
```
Num 이라는 형식의 데이터는 my_classifier/msg에 다음과 같이 선언되어 있습니다.
```
int32 myid
float32 linear
float32 rotation
float32 prob
```
 - linear : 직선 속도를 의미하며 -1.0(후진)~+1.0(전진) 의 값을 가집니다.
 - rotation: 회전 속도를 의미하며 -1.0(우회전)~1.0(좌회전) 의 값을 가집니다.
 - prob: 각 회전량에 대한 확률값을 전달합니다.
 
조이스틱의 1번 버튼 입력에 따라 조이스틱 구동과 자동주행 모드를 변경합니다.
```py
    # Read the most recent button state
    newJoyButtons = [0,0,0,0,0,0,0,0]
    newJoyButtons = deepcopy(data.buttons)
    # Check if button 1(B) is newly set
    if (newJoyButtons[1]==1) and (newJoyButtons[1]!=joyButtons[1]):
        if isAutoMode!= True:
            isAutoMode = True
        else:
            isAutoMode = False
    # Update button state
```
여기서 결정된 주행 모드에 따라 제어 명령으로 변환하게 됩니다.
in line 61~68
```py
    if isAutoMode!= True:
        joy_v = joyAxes[3]
        joy_w = joyAxes[2]
        print "Joy mode: {:.2f} {:.2f} ".format(joy_v, joy_w)
    else:
        joy_v = cmd.linear
        joy_w = cmd.rotation
        print "Auto mode: {:.2f} {:.2f}".format(joy_v, joy_w)
```
###my_classifier 노드
my_classifier노드는 3개의 인자를 전달받습니다.
 - --mean : Caffe에서 생성한 mean binary image파일
  - --prototxt : Caffe의 'deploy' prototxt 모델 정의 파일
   - --model : 훈련된 model 파일
   
   대부분은 pycode안에 있는 rt_classification.py와 유사하고 이미지를 신경망에 넣고 출력하는 부분은 talker() 함수에 들어있습니다.
신경망을 거쳐서 나온 output에 따라서 다음과 같이 제어 명령의 속도와 회전량을 결정하게 됩니다.
```py
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
```

### ROS 빌드하고 실행하기
이제 로봇을 구동할 차례입니다.
저장소의  ros_catkinws폴더로 이동하여 ROS 작업 폴더를 생성합니다.
```sh
cd ~/deep_learning_caffe/ros_catkinws
catkin_make
```
필요한 ROS 패키지가 모두 설치되어 있다면 문제 없이 설치가 완료됩니다.
만약 Joy 노드 관련 패키지가 설치되어 있지 않아서 에러가 발생하는 경우 다음을 실행하고 나서 catkin_make 를 다시 수행합니다.
```sh
$ sudo apt-get install ros-kinetic-joystick-drivers 
```
설치가 완료되면 ros_catkinws 폴더 안에 src 외에 build 폴더가 생성됩니다. 
build 폴더로 이동해서 install 을 진행합니다.
```sh
cd build
make install
```
build 폴더 안에 devel 이라는 폴더가 생성되고 setup.bash 라는 파일이 있습니다. 여기에는 ROS 노드들의 실행 명령어와 실행 Path등이 정의되어 있습니다. 이 파일의 경로를 터미널 Shell 환경을 정의하는 ~/.bashrc 에 등록하면 매전 각 노드들의 전체 경로를 입력하지 않아도 실행할 수 있습니다. 방법은 다음과 같습니다.

```sh
$ echo -e "source ~/deep_learning_caffe/ros_catkinws/devel/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
```
이렇게 입력하면 새로운 터미널을 열때마다 ROS노드들의 전체 경로를 입력하지 않아도 실행이 가능합니다.
4개의 터미널 창을 열고 각 창에서 다음을 순서대로 입력합니다.
```
roscore
rosrun joy joy_node
rosrun my_classifier classifier.py --mean ~/deep_learning_caffe/input/mean.binaryproto --prototxt ~/deep_learning_caffe/caffe_models/caffe_model_1/caffenet_deploy_1.prototxt --model ~/deep_learning_caffe/caffe_models/caffe_model_1/caffe_model_1_iter_3000.caffemodel
rosrun myjoy joycon.py 
```
축하합니다. 이제 여러분이 만든 딥러닝 기반의 로봇이 동작을 시작할 것입니다!


질문이나 의견 혹은 bug 리포트 등은 제 이메일 kyuhyong [at] gmail [dot] com 로 보내주십시오.
감사합니다.

수요일, 07. 2월 2018 작성됨
