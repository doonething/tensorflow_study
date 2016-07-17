ksttps://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet


머신 러닝에서 hello world 는 mnist 를 학습하는 것이다.


MNIST 는 사람이 손으로 쓴  0 - 9 사이의  수를 의미하는 자료들이다.

단순 그림 파일들만 있는 것이 아니라, 각 그림이 실제 어떤 수자인지 표시( label ) 되어 있는 정보도 포함 되어 있다.

여기서, 각 그림이 어떤 수자인지 예측할 수 있도록 훈련 시킬 것이다.

?
In this tutorial, we're going to train a model to look at images and predict what digits they are. Our goal isn't to train a really elaborate model that achieves state-of-the-art performance -- although we'll give you code to do that later! -- but rather to dip a toe into using TensorFlow

이 투토리얼에서 이미지가 어떤 숫자인지 판단하는 모델(?) 을 훈련 시킬 것이다. 

실제로 훌륭한 ( state-of-the-art ) 성능을 내는 정교한 모델을 훈련시키는 것이 아니라
tensorflow 를 맛보기 수준이다. 
실제 정교한 모델은 후에 제시될 것이다. 

매우 간단한 모델(?)인 *softmax regression* 으로 시작할 것이다.


실제 코드는 매우 짧고, 3줄에 모든 흥미로운 것이 들어 있지만, 이를 이해하는 것이 중요하며, 
tensorflow 가 어떻게 동작하는지와 머신러닝의 중요한 개념을 담고 있다.

MNIST Yann LeCun's website 에서 있다.
편의를 위해 자동으로 다운로드하고 설치하는 python code 를 제공한다. 
직접 다운로드 해도 된다.


MNIST 는 세가지로 구성 된다.

55000 개의 훈련 데이터 ( mnist.train )
10000 개의 실험 데이터 ( mnist.test )
5000  개의 검증 데이터 ( mnist.validation )

이렇게 데이터를 분류하는 것은 중요하며, 학습 하지 않은 데이터를 분리하는 것은, 확실히 일반화를 했는지 확인 할 수 있게 한다.

각 데이터는 이미지와 그 이미지가 실제 어떤 수자를 의미하는지  표시하는 정보도 ( label )  있다.
해당 데이터의 의미를 표시하는 정보를 라벨이라고 하겠다.
이미지를 xs 로, 라벨을 ys 로 정한다.

mnist.train.images 는 이미지 데이터를 
mnist.train.labels 가 라벨 정보를 나타낸다.

각 이미지는 28 X 28 로 784 개의 점으로 이루어진다.
MNIST 의 이미지 784 차원의 벡터 공간의 데이터로 볼 수 있다.

784 차원의 벡터로 보는 것은 2차원 정보를 무시하는 것으로 볼 수 있으나, 간단히 하기 위해서다.
후에 다른 튜토리얼에서 2차원 정보도 다루는 예저가 있을 것이다.

mnist.train.images 는 [55000, 784] 로 생겨먹은 텐서이다.
[55000, 784] 는 55000 개의 이미지가 있고 각 이미지는 784 개의 점으로 이루어져 있다는 뜻이다.
이미지의 각 점은 0 부터 1 사이의 수이다.

전 부터 얘기한 라벨은 10개의 비트로 이루어진 벡터로 볼 수 있다.
이 라벨은 10 개의 비트로 이루어진 one-hot 벡터로 볼 수 있다.
one-hot 벡터는 한개의 비트만 1 이고 나머지는 모두 0 이다.
mnist 는 라벨에서는 9 개가 0이고, 나머지 한개가 1 이다.



## softmax regressions

이미지마다 확률을 줄 수 있다. 
예를 들어, 훈련시킬 모델이 9의 이미지를 80% 확률로 9로 볼 수 있고, 5% 의 확률로 8로 볼 수 있다.
나머지 수자에는 작은 확률을 줄 수 있다.

이 형태가 소프트맥스의 형식이다.

This is a classic case where a softmax regression is a natural, simple model. If you want to assign probabilities to an object being one of several different things, softmax is the thing to do. Even later on, when we train more sophisticated models, the final step will be a layer of softmax.

소프트 맥스 는 두 단계로 이루어 진다.

A softmax regression has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities.
첫째 입력에 대한 증거( evidence ) 를 특정 클래스(?) 에 더해간다. ( add up )
둘째 이 증거(?) 를 확률로 변경한다.

