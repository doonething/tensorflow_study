ksttps://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

머신 러닝에서 hello world 는 mnist 를 학습하는 것이다.


MNIST 는 사람이 손으로 쓴  0 - 9 사이의  수를 의미하는 자료들이다.

단순 그림 파일들만 있는 것이 아니라, 각 그림이 실제 어떤 수자인지 표시( label ) 되어 있는 정보도 포함 되어 있다.

여기서, 각 그림이 어떤 수자인지 예측할 수 있도록 훈련 시킬 것이다.

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

이런 형태에 고전적으로 소프트맥스 회귀 분석이 적용된다.
다양한 경우 수 중 하나에 여러 확률을 정의 하고 싶다면, 소프트맥스가 해준다.
나중에 복잡한 모델을 적용해도 마지막 단계는 소프트맥스일 것이다.


소프트 맥스 는 두 단계로 이루어 진다.
첫째, 입력 값이 특정 클래스 속한 다는 증거( evidence ) 를 더해간다.
둘째, 이 증거(?) 를 확률로 변경한다.

이미지가 특정 클래스에 속한 다는 모든 증거를 더하기 위해서는 각 이미지의 점의 채도에 가중치를 부여(곱하여)하여 모두 더한다.
가중치 값이 음수면, 특정 클래스에 속하지 않는다는 강한 증거이며, 양수이면 특정 클래스에 속한다는 것이다.
( 여기서 클래스는 10개의 라벨중 하나를 의미한다. )

다음 그림은 특정 모델이 학습한 가중치를 의미한다.
빨간색은 음수 가중치, 파란색은 양수 가중치를 의미한다.

![alt text](https://www.tensorflow.org/versions/r0.9/images/softmax-weights.png)

그리고 추가적인 증거로 불리는 bias 를 더한다.

Basically, we want to be able to say that some things are more likely independent of the input. 

수식

evidence(i) = sum ( W(i,j) * x(j) , iter=j ) + b(i)

W( i , ? ) 가 가중치  b(i) 가 특정 클래스 i 에 대한 바이어스 
j 가  x 의 이미지의 각 픽셀의 인덱스를 의미한다. 

( 0<= j <= 784 ) 

그리고, 더해진 증거를 예측된 확률로 변경하기 위해 softmax 함수를 사용한다.

  y = softmax ( evidence )

이 소프트맥스 함수가 활성함수(activation) 혹은 연결(link)함수 이며,
선형 함수를 10 가지 종류의 확률 분포 함수로 변환한다.


  softmax ( x )  = normalize ( exp (x) ) 


좀더 , 자세히
                          exp ( x(i) ) 
  softmax ( x(i) )  = ——————————————————
                     sum ( exp(x(j) ), j )

이 소프트맥스 의미는 입력을 지수적( exponentiating ) 으로 변경하고, 정규화( normalizing ) 한거로 볼 수 있다.

The exponentiation means that one more unit of evidence increases the weight given to any hypothesis multiplicatively. 
And conversely, having one less unit of evidence means that a hypothesis gets a fraction of its earlier weight.

지수적으로 변경한다는 의미는 ( exponentaition ) 
1 보다 큰 증거는 주어진 가설(?)에 따라 지수 함수적으로 가중치를 증가 시키고,
반대로 1 보다 작은 증거는 원래 가중치 보다 작아진다.

여기서 증거는 한 이미지로 부터 구한  Wx + b 이다.
가설은 가중치를 변경시키는 방법으로 생각할 수 있다.

가설(hypothesis) 는 0 또는 음수의 가중치는 가질 수  없다.
정규화( normalize ) 를 하여, 모두 더한 값이 1이 되도록 한다.
( 자세한 사항은 Michael Nielsen 의 책을 참고하시오 )


아래 그림과 같은 형태로 표시할 수 있다.

![alt text](https://www.tensorflow.org/versions/r0.9/images/softmax-regression-scalargraph.png)

그림 ( 가중함에 softmax )

그림 ( 매트릭스 벡터 곱 )

식  y = softmax ( Wx + b )



## 회귀 구현  Implementing the Regression

행렬  벡터 계산을 위해  Numpy 를 사용한다.
Numpy 에는 고 효율성을 위해 다른 컴퓨터 언어로 만들어진 부분도 있다. 
하지만, python 과 이 다른 언어 사이의 교환 과정(?) 은 오버헤드가 크다.
이 오버헤드는 GPU, 데이터 교환 비용이 큰 분산환경에서 좋지 않다.

tensorflow 는 이 오버헤드의 우회 방법을 가지고 있다.
각각의 고비용 작업을 하나하나 python 에서 실행하지 않고,
전체 작업을 한꺼번에 python 외부에서 실행하도록 한다.

tensorflow 를 사용하기 위해서 import 해야 한다.

```python
 import tensorflow as tf
```

```python
 x = tf.placeholder (tf.float32, [ None, 784 ] )
```

여기서 x 는 특정 값을 가지는 것이 아니고, 값을 받을 수 있는 placeholder 이며, tensorflow 가 동작할 때 이 placeholder에 입력해야 한다.
이미지 개수에 상관없이 입력 받기 위해 None 사용

가중치와 바이어스를 위해 placeholder 를 사용할 수 도 있으나, 더 좋은 Variable를 사용함.
이 Variable 은 변경 될 수 있는 텐서이며, tensorflow 가 동작 중에도 계속 사용할 수 있다.
머신 러닝에서 모델의 파라미터 ( 가중치, 바이어스 ) 는 Variable 이 일반적으로 사용 된다.


```python
 W = tf.Variable( tf.zeros([784,10] )
 b = tf.Variable( tf.zeros([10] )
```

위 코드로 Variable 을 생성했으며, W, b 모두 0으로 초기화 함
Since we are going to learn W and b, it doesn't matter very much what they initially are.
W, b 를 학습하기 시작하면, 어떤 값으로 초기화 되는 상관없다.

784 차원의 벡터를 곱하고 결과를 10차원 벡터를 얻기 위해, W 의 형태는 [784, 10] 이다. 
W * x 가 10차원 벡터이므로 b 도 10차원 벡터이다.

따라서 다음 파이선 코드처럼 구현한다.

```python
 y = tf.nn.softmax( tf.matmul(x,W) + b )  
```

  x = [ none, 784 ], w=[784, 10 ]
  y = [ none, 10 ]

첫째, x 에 W 를 곱한 코드가 tf.matmul(x,W ) 이다. 
This is flipped from when we multiplied them in our equation, where we had Wx
, as a small trick to deal with x being a 2D tensor with multiple inputs.
원래 W 에 x 를 곱하지만, x가 2D 텐서라서 살짝(?) 바꾼거 같다.
그리고 b 를 더한 다음 softmax 를 사용했다.

이것이다. 설정 관련 몇 줄 후에 , 이 한줄로 모델을 정의한 것이다.
That isn't because TensorFlow is designed to make a softmax regression particularly easy: 
( 무슨 소리인지 모름 ) 
tensorflow 가 소프트맥스 회귀 분석을 위해 디자인 된 것이 아니다. ( ? )
it's just a very flexible way to describe many kinds of numerical computations, from machine learning models to physics simulations. 
이렇게 하는 것은 머신러닝에서 부터 물리적 시뮬레이션까지 적용 되는 유연한 계산 방법이다.
한 번 정의 된 이 모델은, 다른 장비에서 실행 될 수 있다. cpu , gpu , phone 에서도 실행 된다.


## Trainng 훈련
모델을 훈련 시키기 위해서 무엇이 좋은 것인지 정의할 필요가 있다.
하지만, 머신 러닝에서는 정통적으로 cost 혹은 loss 라고 불리는 함수를 정의하여 무엇이 나쁜지 정의 한다.
그리고 cost, loss 함수가 최소가 되도록 한다.

일반적으로 , cross entropy 함수가 좋다. 
놀랍게도, 정보 이론에서, 정보 압축을 위해 탄생된 이 cross entropy 는, 게임에서 부터 머신러닝 까지 많은 다른 분야에서 사용 된다.

식   H(y, y’)  = - sum ( y’(i) log(y(i) ), i )

y 는 모델이 예측해야할 확률 분포이고, y’ 는 현재 모델에 의해 계산된 확률 분포이다.
대충, 예측한 값이 얼마나 틀린 것인지 보여준다고 할 수 이다. 
이 cross entropy 는 이 투토리얼의 범위를 벗어나지만, 이해하는 것은 가치있다.

외부 url 링크 http://colah.github.io/posts/2015-09-Visual-Information/

이 cross entropy 를 구현하기 위해, 먼저 실제 모델이 예측해야하는 값 ( 라벨) 을 담을 placeholder 를 만들어야 한다.

```python
y_ = tf.placeholder(tf.float32, [None, 10])
```

그럼 다음 코드로 cross entropy 를 계산할 수 있다.

```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```

tf.log 로 각 y 의 로그 값을 계산 한다.
그리고 y 값에 맞는  라벨값 y_ 을 곱한다.
그리고 tf.reduce_sum () 으로 , y*y_ 로 부터 만들어진 행렬에서  두 번째 요소를 모두 더한다.

y*y_ 는 행렬 곱이 아니다.
y, y_ 모두  [ none, 10 ] 이고, 
y*y_ 는 행렬 곱이 아니라, 동일 행렬이므로 같은 자리 원소값만 곱한다. 
y*y_ 도  [ none, 10 ] 이다.
즉 , y*y_ = yy 일때,   yy(i,j) = y(i,j) * y_(i,j) 이다.

두 번째 요소라는 의미는 reduce_sum 의 두 번째 인자인  reduction_indices=[1] 에서 정의 된다.

그리고 reduce_mean 으로 평균을 구한다


그리고, 이제 모든 계산 그래프가 만들어 졌기 때문에, 아래 코드로 백프로파게이션을 수행하여 cross entropy 가 최소값이 되도록 훈련한다.

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

학습률은 0.5 이며, tensorflow 는 다른 많은 알고리즘을 제공한다.

여기서, tensorflow 는 관련된 모든 계산 과정을 그래프로 모두 구성하며 백프로파게이션과 그라디언 디센트를 구현한다.
이 모든 계산 과정은 여러 변수들을 계산하여 하나의 스칼라 값인 cost 를 구한다.


tensorflow 에서 variable 를 사용했다면, tensorflow .initialize_all_variables() 함수를 꼭 호출해야 한다.

```python
init = tf.initialize_all_variables()
```

위에서 정의한 모든 계산 작업의 그래프는 tensorflow. Session() 이용해야지만, 실행할 수 있다.

```python
sess = tf.Session()
sess.run(init)
```

1000 번을 학습하는 코드는 아래와 같다.

```python
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

한 번에 100개의 훈련 데이터를 가져오며,
x, y_ 에 값을 할당한다.

이 예제는 추측통계학적(stochastic)인 훈련 방법이다. 실제로 stochastic 그라디언트 디센트 사용함
즉, 실제 데이터의 일부만 사용했다는 것이다.
이상적으로 모든 데이터로 훈련하는 것이 좋지만 비용이 크다.


## 제대로 학습 했는가? Evaluating Our Model

