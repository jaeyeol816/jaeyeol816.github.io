---
title:  "[딥러닝1] 딥러닝에서 학습은 어떻게 일어날까?"
categories:
  - deep_learning
use_math: true
---

## 들어가기에 앞서...

> - 이 글은 딥러닝을 처음 접하시는 분들을 위해 개념적이고 추상적인 내용만 담았습니다. 본격적인 공부에 앞서 기본적인 내용을 이해할 수 있도록 도와드리겠습니다. 다음 포스팅에서 딥러닝의 수학적인 이해를 다루겠습니다.
> - 일부 용어에 있어서 한국어로 번역된 것보다 영어를 사용하는 것을 지향합니다. 이해해 주시면 감사하겠습니다!
> - 잘못된 내용이 있으면 언제든 피드백 주세요! 빠르게 고치도록 하겠습니다 🥰

<br>

# 목차

순서대로 읽으시는 것을 추천드립니다!
1. '인공 신경망'과 '신경망' &#160;&#160; [👉바로가기](#1-인공-신경망과-사람의-신경망)
2. 퍼셉트론 (Weight & Activation)  &#160;&#160; [👉바로가기](#2-퍼셉트론-weight--activation)
3. 다중 퍼셉트론  &#160;&#160; [👉바로가기](#3-다중-퍼셉트론)
4. Layer  &#160;&#160; [👉바로가기](#4-unit과-layer)
5. 학습이란? Weight를 찾아가는 과정이다! &#160;&#160; [👉바로가기](#5-학습이란-weight를-조정해-가는-과정이다)
6. Forward-propagation과 Loss &#160;&#160; [👉바로가기](#6-forward-propagation과-loss)
7. Gradient와 Back-propagation &#160;&#160; [👉바로가기](#7-gradient와-back-propagation)
8. 정리하자면... &#160;&#160; [👉바로가기](#8-정리하자면)


<br><br>

# 1. '인공 신경망'과 '사람의 신경망'

&#160;딥러닝 구조는 사람의 뇌 속의 신경 세포의 구조를 모델화 한 것이라고 합니다. 딥러닝 구조를 흔히 '인공 신경망(artifical neural network)' 라고도 하죠. <br>
&#160;그렇다면 둘은 어떤 공통점이 있는 것일까요? 아래 그림은 사람의 신경망과 딥러닝의 신경망을 나타낸 것입니다.


<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture1.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진1) 사람과 딥러닝의 neural network 
  </figcaption>
</figure>

&#160;사람의 신경세포(뉴런)는 복잡하게 서로 연결되어 있습니다. 사람한테 외부 자극(정보)이 들어오면 전기 신호로 변환된 후 전기 신호가 수많은 뉴런 사이를 이동하게 됩니다. 이때 한 뉴런에서 출력된 전기 신호는 다른 뉴런의 입력으로 들어가게 돼요. <br>
&#160; 딥러닝에서도 마찬가지입니다. 특정 행동을 하는 뉴런 하나하나가 서로 복잡하게 연결되어 있습니다. 마찬가지로 한 뉴런에서 계산된 출력값이 다른 뉴런의 입력값이 되죠. 여기서 세포, 즉 뉴런 하나가 어떤 역할을 하는지는 다음 파트에서 설명 하겠습니다.

<br><br>

# 2. 퍼셉트론 (Weight & Activation)

&#160;딥러닝 모델을 흔히 MLP (Multi Layer Perceptron) 이라고 합니다. 퍼셉트론이 여러 층으로 이루어진 구조라고 해석할 수 있습니다. 딥러닝에서 퍼셉트론에 대응하는 사람의 뉴런(신경세포)에서의 연산을 간단히 알아봅시다.
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture2.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진2) 사람의 뉴런(신경세포)
  </figcaption>
</figure>
&#160;생명과학 시간에 수상돌기, 축삭돌기, 가지돌기.. 등등 재미있는 단어를 들어본 적이 있으실 텐데요, 신경 세포의 구조를 설명하는 단어입니다. 신경 세포는 (사진2)의 Dendrites(수상돌기)를 통해 전달받은 신호가 특정 임계점 이상이 올때 Axon(축삭돌기)를 통해 다음 신경 세포로 전달합니다. 딥러닝 구조에서 뉴런의 역할도 이것과 비슷하다고 할 수 있습니다. 
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture3.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진3) Perceptron의 개념적 이해
  </figcaption>
</figure>
&#160;퍼셉트론은 신경세포의 동작 과정을 통계학적으로 모델링한 알고리즘입니다. (사진3)을 보며 이해해 봅시다. 하나의 퍼셉트론은 여러 Input을 가질 수 있습니다. (사진3)에서 🍎, 🍏, 🍊는 들어오는 Input을 나타냅니다. 이러한 Input들에게 첫번째로 Weighted Sum(가중합)을 구하는 연산이 적용됩니다. 🍎(첫번째 input)에는 $ w_1 $, 🍏(두번째 input)에는 $ w_2 $, 🍊(세번째 input)에는 $ w_3 $ 이 곱해진 후 서로 더해지게 되죠. <br>
&#160;구해진 가중합을 $ Z $라고 합시다. 이후 Activation Function(활성 함수)를 거치게 되는데, $Z$값이 특정 Threshold(임계값) 이상이면 1이 출력되고, 이하이면 0을 출력하게 됩니다. <br>
&#160;실제 딥러닝에서는 Activation Function으로 위와 같은 계단 함수 말고 Sigmoid Function, ReLU Function등이 쓰입니다. 다음 포스팅에서 소개해 드리도록 하죠.

&#160;뉴런으로 들어온 정보가 종합(Linear Function 적용)된 후, 임계점 이상이면(Activation Function 적용) 1을 다음 뉴런으로 넘긴다는 점이 사람의 뉴런과 비슷하게 느껴지시지 않나요? <br>
&#160;퍼셉트론 한 개를 거쳤을 때 일어나는 일은 이해를 하셨을 겁니다. 하나의 퍼셉트론만 보셨을 때는 Regression(회귀)를 한 후 임계치에 따라 판단을 하는 과정입니다. 하지만 퍼셉트론이 여러 개 모였을 때에는 놀라운 결과를 만들어 낼 수 있습니다. 다음 절을 참고해주세요!

<br><br>

# 3. 다중 퍼셉트론
시간이 없으신 분들은 이번 단원은 건너뛰어도 좋아요.<br>
&#160;퍼셉트론을 하나만 사용했을 때는 좌표명면 상의 데이터를 직선으로밖에 분리하지 못합니다. 아래 그림(사진4)처럼 말이죠 (2차원 데이터의 선형 분리 예시).
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture4.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진4) 선형적으로 데이터 분리하기
  </figcaption>
</figure>
&#160;입력값(2차원 좌표)에 따른 결과값(색상)을 직선 형태로밖에 구분하지 못합니다. (예를 들어 직선보다 왼쪽-위에 있으면 파랑, 직선보다 오른쪽-아래에 있으면 빨강) <br>
&#160;하지만 한 퍼셉트론에 뒤이어 다른 퍼셉트론을 붙이면, 즉 층(layer)을 더하면 비선형 문제를 풀 수 있습니다. 
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture5.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진5) 단일 층을 사용할 경우 직선 모양의 Decision Boundary
  </figcaption>
</figure>
&#160;위 그림은 AND, NAND, OR 문제입니다. AND층은 두 입력값이 TRUE일 때 TRUE를 출력하며, NAND(NOT AND)층은 두 입력값이 FALSE일 때 TRUE를 출력합니다. OR층은 두 입력값 중 하나라도 TRUE이면 TRUE를 출력합니다. <br>
&#160;이때 입력값 $ x_1 $, $ x_2 $에 대한 결과값을 색상으로 나타내 보았습니다. 결과값이 TRUE인 영역과 결과값이 FALSE인 영역이 "직선"으로 구분 가능함을 알 수 있습니다. 즉, Decision Boundary가 직선입니다.
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture6.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진6) XOR 문제
  </figcaption>
</figure>
&#160;XOR연산은 두 가지 입력 중 하나만 TRUE일 때만 TRUE이고 아니라면 FALSE인 연산입니다. (사진6)에 표현된 XOR연산의 Decision Boundary를 보면 직선이 아니라 곡선입니다. XOR의 Decision Boundary는 선형적으로 표현이 불가능하기 때문에 하나의 퍼셉트론 층으로는 표현할 수 없습니다. <br>
&#160;하지만 첫 층(layer)에 NAND와 OR를 두고 두번째 층(layer)에 AND을 두면서 비선형적인 결과를 만들어내는 XOR 연산을 표현할 수 있게 되었습니다.

&#160;퍼셉트론 한 층만으로는 선형 문제밖에 풀지 못했지만, 두 층을 두었더니 더 복잡한 문제를 풀 수 있었습니다. 이처럼 여러 층의 퍼셉트론을 모아 하나의 모델을 구성한 것을 인공신경망(Artificial Neural Network)라고 합니다. 퍼셉트론이 신경세포를 본 뜬 것이니 '신경망(Neural Network)'이라는 단어를 선택한 것입니다.


<br><br>

# 4. Unit과 Layer
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture7.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진7) Unit과 Layer
  </figcaption>
</figure>

**Unit** <br>
&#160;이제 딥러닝에서 쓰이는 용어로 넘어가도록 합시다. 지금까지 '뉴런(neuron)', '퍼셉트론(perceptron)'이라고 불렀던 것을 딥러닝에서는 '노드(node)' 또는 '유닛(unit)'이라고 불러요. 저는 앞으로 유닛(unit)이라는 표현을 사용하도록 하겠습니다.

**Layer**<br>
&#160;이전 단원에서 다중 퍼셉트론을 설명할 때 NAND유닛과 OR유닛이 하나의 '층'을 이루어고, AND유닛이 하나의 '층'을 이루었습니다. 이러한 '층'을 '레이어(layer)'라는 용어로 부릅니다. 
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture8.png" width="200px"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진8) Layer의 종류
  </figcaption>
</figure>

**Input Layer, Hidden Layer, Output Layer**<br>
&#160;Layer는 크게 세 종류로 나눌 수 있습니다. '입력층(Input Layer)', '은닉층(Hidden Layer)', '출력층(Output Layer)'으로 말이죠. Input Layer는 네트워크에 들어오는 데이터 그 자체를 지칭합니다. <br>
&#160;네트워크로 특정 사람의 '[눈 크기, 코 높이, 머리 색상]' 과 같은 정보가 들어오면 네트워크가 '호감' 또는 '비호감'을 판단한다고 칩시다. 이때 들어오는 데이터 (예를 들어 [3cm, 1cm, 갈색])가 그 자체로써 Input Layer를 의미합니다. 이때 데이터가 3개 원소로 이루어져 있으니 Input Layer의 Unit의 개수는 3개입니다. 최종적으로 '호감' 또는 '비호감'이라는 데이터가 출력되는 Layer을 Output Layer라고 합니다. Output Layer는 마지막 Layer가 되는 거죠. <br>
&#160;Input Layer도 아니고 Output Layer도 아닌 모든 Layer들을 'Hidden Layer'라고 합니다. 중간 Layer의 이름에 Hidden이라는 용어가 들어간 이유은 무엇일까요? Hidden Layer으로 입력되거나 Hidden Layer에서 출력되는 값은 사람의 입장에서 의미있게 해석되는 값이 아니기 때문에 '감춰졌다'라는 표현을 사용합니다. 이 값들이 최종 Output Layer의 결과물을 만드는데 영향은 주지만, 값 하나하나가 의미를 갖고 있지 않습니다.

**Layer간 연결의 특징**<br>
&#160;특정 N번째 Layer가 있다고 칩시다. N번째 Layer의 모든 특정 Unit의 출력은 N+1번째 Layer의 모든 Unit으로 전달된다는 특징이 있습니다. 다르게 말해서, N번째 Layer의 개별 Unit 하나하나는 N-1번째 Layer의 모든 Unit으로부터 Input값을 받는다고 할 수 있죠! (사진7과 사진8을 세심히 관찰해 보시면 무슨 말인지 알 수 있을 거에요.)<br>
&#160; 한 Unit에서는 이전 Layer의 모든 Unit으로부터 정보를 얻어와 계산(Linear Function + Activation) 을 한 후, 그 결과값을 다음 Layer의 모든 Unit으로 전달합니다.

**Layer의 개수** <br>
&#160;일반적으로 Layer의 개수를 셀 때 input layer의 개수는 포함시키지 않아요. 사실 input layer는 연산을 하는 것이 아니라 네트워크에 주어진 데이터 그 자체이기 때문이죠. 따라서 (사진7), (사진8)의 경우 Layer 개수는 5개라고 할 수 있습니다.

<br><br>

# 5. 학습이란? Weight를 조정해 가는 과정이다!

&#160;Neural Network를 하나의 "함수"라고 생각해 봅시다. 우리가 이 함수에 데이터를 입력하면, 함수는 예측값을 출력합니다. 이때 예측값을 정확하게, 즉 실제값과 비슷하게 출력하도록 만드는 것이 우리의 목표입니다.
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture9.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진9) 학습이 반복됨에 따라 weight값들이 업데이트됨
  </figcaption>
</figure>
&#160;데이터가 'layer를 통과하는 것'은 '각 unit에 있는 weight가 곱해진 후 activation function을 통과하는 것'과 마찬가지 입니다. 그러므로 정해진 구조의 Neural Network의 아웃풋을 결정하는 것은 각 Unit의 weight입니다. 따라서 '학습'이라는 것은 최적의 예측값을 만들도록 각 Unit의 weight들이 업데이트되는 과정입니다. 학습이 진행됨에 따라 weight는 점차 변화합니다. <br>
> &#160;믈론 weight뿐 아니라 각 Unit에 linear function부분에 존재하는 bias도 업데이트 하지만, 이는 다음 포스팅에서 자세히 다뤄 보도록 하겠습니다. (이번 포스팅에서는 weight에 대해서만 언급하겠습니다)

&#160;그렇다면 weight는 어떻게 업데이트 되는 것일까요? 매 반복 주기(Iteration)마다 input데이터에 대한 예측값이 출력되고, 예측값에 대한 오차(loss)를 계산한 후, 현재 weight들에 대한 loss의 "변화율"에 기반하여 weight들을 업데이트해나갑니다. 이러한 반복이 진행되면 진행될수록 weight들은 더 낮은 loss를 갖는 예측값을 출력하도록 변화하게 됩니다. 어렵게 느껴지시나요? 이후 내용에서 설명이 이어집니다.

&#160;Deep Learning이라는 용어의 뜻도 어렵게 이해할 필요가 없어요. "Deep" 하다는 것은 layer가 여러 겹 있다는 뜻이고, "Learning" 은 그 상태에서 각 layer의 weight를 업데이트해가며 학습을 한다는 것입니다!

<br><br>

# 6. Forward-propagation과 Loss

&#160;[5단원](#5-학습이란-weight를-조정해-가는-과정이다)에서 딥러닝에서 '학습'이라는 것이 무엇인지 배웠습니다. 이번 단원부터 학습이 일어나는 과정을 한가지 예시와 함께 조금 더 자세히 살펴보도록 해요.

&#160;연예인의 '키', '몸무게', '눈 크기', '코 높이' 를 갖고 '호감 여부'를 예측하는 모델을 만들고 싶습니다. 이 때 학습을 위한 데이터셋은 아래와 같이 주어졌습니다. 

<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture10.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진10) 데이터셋 예시
  </figcaption>
</figure>

&#160;이 데이터에서 초록색 부분은 input입니다. &#160;Input값을 neural network에 넣으면 여러 Layer를 차례대로 통과하며, 마지막 Layer에서는 '호감일 확률'이라는 예측값 출력합니다. &#160; 그런 다음 우리는 이 예측값과 ground truth값 사이의 오차를 계산합니다. 이 과정을 'Forward Propagation'이라고 하죠.

&#160;이 경우 input vector의 크기가 4이므로 input layer의 크기는 4입니다. 첫번째 hidden layer의 모든 unit으로 input vector가 들어가게 됩니다.

<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture11.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진11) 연예인A의 데이터가 첫번째 hidden layer까지 투과한 상황
  </figcaption>
</figure>

&#160; 첫번째 hidden layer의 각 unit는 초기화된 weight가 다르기 때문에 저마다 다른 output을 출력합니다. 이 값들은 두번째 hidden layer의 모든 unit에 들어가게 됩니다.

&#160;두번째 hidden layer의 unit들도 마찬가지로 저마다 첫 hidden layer의 output을 갖고 각자의 output 을 만들어 냅니다. 

<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture12.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진12) 연예인A의 데이터가 output layer까지 투과한 상황 (ground truth와 비교)
  </figcaption>
</figure>

&#160;이러한 두번째 hidden layer의 output은 마지막 layer의 input이 되며, 마지막 layer는 최종적인 '호감도가 1일 확률'을 출력합니다.

&#160;학습 데이터셋에 대해서 우리는 ground truth, 즉 정답을 알고 있습니다. 결국, 방금 설명된 forward propagation의 output이 정답과 얼마나 가까운지 측정할 수 있다는 뜻이기도 하죠.

> &#160;참고. 이렇게 학습 데이터셋의 정답을 알고 있는 형태의 머신 러닝을 'Supervised Learning'이라고 합니다.

&#160; forward propagation으로부터 출력된 값을 $ \hat{y} $ 이라고 하고, ground truth 값을 $ y $ 이라고 합시다. 이 때, 우리는 아래과 같은 공식으로 loss를 구합니다.

$$ L(\hat{y}, y) = -(y\log(\hat{y}) + (1-y)\log(1-\hat{y})) $$

&#160;위 수식을 살펴보면, ground truth($ y $)가 1인 데이터에 대해서는, 수식의 오른쪽 부분은 0이 되고 왼쪽 부분만 살아남게 되어 loss는 $ -\log(\hat{y}) $ 이 됩니다. 이때, 로그함수의 모양을 생각해 보면 $ \hat{y} $가 작을수록 loss는 커지게 됩니다. 즉 $ y $가 1일때는  $ \hat{y} $도 1에 가까워야 loss(오차)가 작아진다는 것을 잘 나타내므로 이 수식이 말이 되는 것을 확인할 수 있죠. $ y $가 0일 때를 생각해 봐도 이 수식의 타당성을 이해 할 수 있습니다.

&#160;(사진12)를 봅시다. 연예인A 데이터를 neural network에 forward propagate시킨 모습입니다. Neural network는 0.873이라는 $ \hat{y} $ 값('호감'일 확률)을 출력했습니다. 연예인A는 '호감'에 속하므로 $ y $값은 1이며, 위 수식에 넣어 계산한 loss는 -0.059 가 됩니다. 나름 잘(?) 맞췄으므로 상당히 작은 loss가 계산된 것을 확인하실 수 있습니다.

&#160;그렇다면 이렇게 forward-propagation을 통해 loss를 구하는 것과, weight를 업데이트하는 '학습'과는 어떤 관계가 있는 걸까요? [다음 단원](#7-gradient와-back-propagation)에서는 loss가 계산된 이후 어떻게 weight를 update되는 것인지 이해해 보도록 하겠습니다.

<br><br>

# 7. Gradient와 Back-propagation

&#160;[5단원](#5-학습이란-weight를-조정해-가는-과정이다)에서 딥러닝에서 '학습'이란 loss를 최소화 하기 위해 weight를 업데이트해 가는 과정이라고 배웠습니다. 그렇다면 weight는 어떻게 업데이트를 하는 것일까요? 이 방법을 배우기 위해서 우선 weight에 대한 loss의 그래프를 살펴볼 필요가 있습니다.

**Convex Function**

<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture13.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진13) weight에 대한 loss의 그래프
  </figcaption>
</figure>

&#160;볼록 함수(convex function)에 대해 들어 보셨나요? 우리가 가장 잘 convex function은 이차함수 입니다. 함수가 최솟값을 가지게 하는 특정 x값이 존재하는 것이 특징입니다.

&#160;우리의 목표는 loss를 최소로 하는 weight를 찾는 것입니다. Weight에 대한 loss의 함수가 (사진13)과 같이 convex function이라고 해 봅시다.<br>
&#160;그렇다면 (사진13)의 왼쪽 그림인 2차원 함수의 경우 loss값이 최소가 되도록 하는 weight값이 존재하게 되며, (사진13)의 오른쪽 그림인 3차원 함수의 경우 loss값이 최소가 되도록 하는 weight값 쌍이 존재하게 됩니다.

**Optimization 과정**

<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture14.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진14) w값에 기울기를 뺌으로써 optimal point의 w값으로 이동하도록 하는 과정
  </figcaption>
</figure>


&#160;그렇다면 loss가 최소가 되기 위해서는 weight를 어떻게 움직이는 것이 좋을까요? (사진14)의 왼쪽 2차원 그래프를 봅시다. $ w_1 $ 이 0일 때 loss가 최솟값이 되죠. $ w_1 $이 0보다 작은 상황이면, $ w_1 $ 값을 오른쪽(더 커지게)으로 업데이트하고 $ w_1 $ 이 0보다 큰 상황이라면 왼쪽으로(더 작아지게) 업데이트하면 되겠죠? 

&#160;이것을 구현하려면 어떻게 하면 될까요? "기울기(의 상수곱)를 빼는 과정"을 통해 $ w_1 $ 값을 볼록점의 $ w_1 $ 값으로 모이게 할 수 있습니다. 왜 기울기를 빼는 것이 $ w_1 $ 을 볼록점의 $ w_1 $ 값(위 그림에서 0)으로 모이게 하는지 알아봅시다. <br>
&#160;$ w_1 $ 값이 0보다 작다면, 양수를 더해야 0으로 모이게 할 수 있습니다. 이때 $ w_1 $ 에서 loss함수의 기울기가 음수이기 때문에 이 '음수 상태인 기울기'를 빼는 것이 양수를 더하는 것과 같은 효과를 얻습니다. 
&#160;$ w_1 $ 값이 0보다 크다면, 양수를 빼야 0으로 모이게 할 수 있습니다. 이때 $ w_1 $ 에서 loss함수의 기울기가 양수기 때문에 이 기울기를 빼는 것으로 $ w_1 $ 값을 0(볼록점의 $ w_1 $ 값)으로 모이게 할 수 있습니다.

&#160;이렇게 loss가 최소가 되도록 weight값을 옮겨 가는 과정을 "optimization" 이라고 하며, 그 때의 좌표를 "optimal point"라고 합니다.

&#160;(사진14)의 왼쪽 그림은 neural network에서 weight가 하나일 때 loss의 그래프를 나타낸 것이라면, (사진14)의 오른쪽 그림은 weight가 두 개일 때 loss의 그래프를 나타낸 것입니다. 이 때 $ w_1 $ 도 optimal point으로 옮기고 $ w_2 $ 도 optimal point으로 옮길 필요성이 있습니다. $ w_1 $ 값도 기울기를 빼고 $ w_2 $ 도 마찬가지로 기울기를 빼는 과정을 통해 $ w_1 $ 와 $ w_2 $ 를 각각 optimal point으로 수렴시키면 됩니다.

&#160;실제로 neural network에서의 weight의 개수는 한개도 아니고 두개도 아닙니다. 각 layer마다, 그리고 각 unit마다 여러 개의 weight가 존재하며, 때로는 수천개가 되기도 합니다. 2차원, 3차원의 그래프는 그릴 수 있지만 수천 차원의 그래프는 그릴 수 없습니다. 하지만 그 때도 이렇게 각 weight마다 "본인에 대한 loss의 변화율"을 뺌으로써 optimize를 수행할 수 있다는 것은 똑같이 적용됩니다.

**Local Optima**

&#160;'기울기를 빼는' optimization과정을 아무때나 할 수 없습니다. 우리가 처음에 전제로 했던 볼록함수(convex function)여야 한다는 것이 필요조건입니다. 이렇게 기울기를 빼가면서 optimization을 했더니 loss의 전체 최솟값이 아닌 부분 최솟값에 수렴하게 만들 위험도 있습니다. 아래 (사진15)처럼 말이죠. 이렇게 함수 전체 구간에서의 최솟값이 아닌 부분 구간에서의 최솟값으로 optimize되는 현상을 "local optima problem"이라고 합니다.

<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture15.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진15) Local Optimum과 Global Optimum
  </figcaption>
</figure>

&#160;하지만 현재로써 걱정할 필요는 없습니다. weight의 개수가 많아지면 결국에는 convex point가 하나 존재하게 된다는 것이 알려져 있기 때문이죠.


**Back-propagation (역전파)**

&#160;그렇다면 neural network의 각각의 weight에 대한 loss의 변화율은 어떻게 구하면 될까요? 

&#160;Neural network의 구조부터 생각해 봅시다. weight는 layer의 unit마다 존재합니다. 이 상황에서 모든 layer의 weight들 각각을 "자기 자신에 대한 loss의 변화율"을 구해야 하는 것이 우리의 과제입니다. "특정 weight에 대한 loss의 변화율"을 이제부터 "gradient"라는 용어로 부르기로 하죠.

<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture16.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진16) Back propagation의 이해
  </figcaption>
</figure>

&#160;각 weight의 gradient를 구하기 위해 우리는 마지막 layer의 weight부터 "역방향"으로 미분을 해 나갑니다. 여기서 핵심은 "합성함수의 미분" 입니다. <br>
&#160;(사진16)과 같은 neural network를 합성 함수라고 생각해봐요. 인풋 데이터를 주었을 때 주황색 unit 함수, 노란색 unit 함수, 초록색 unit 함수, 파란색 unit 함수 순서대로 거친 후 ground truth와 비교하는 함수를 거쳐 loss가 계산된다고 해 봅시다 (실제로는 색칠된 unit 뿐 아니라 모든 unit을 거치는데 이해를 위해 색칠한 unit에 대해서만 언급하겠습니다.) 

&#160;첫 번째로, 합성함수에서 가장 마지막 함수라고 할 수 있는 loss function의 변화율을 구합니다. Loss function은 neural network의 output를 loss로 매핑하는 함수이며, 이 함수의 미분을 통해 "output에 대한 loss"의 변화율, 즉 $ dLoss \over dOutput $ 이라고 할 수 있습니다.

&#160;그 다음 단계는, 파란색 unit의 weight인 $ w_🍎 $ 에 대한 loss의 변화율을 구하는 것이 목표입니다. 파란색 unit에 존재하는 linear함수와 activation함수는 미분이 가능하므로 $ dOutput \over dw_🍎 $ 는 충분히 한번에 구할 수 있습니다. 하지만 우리가 구하고 싶은 것은 $ dLoss \over dw_🍎 $ 입니다. 이것은 어떻게 구할 수 있을까요? 이전 단계(보라색)에서 구했던 $ dLoss \over dOutput $ 에다가 이번 단계(파란색)에서 구한 $ dOutput \over dw_🍎 $ 를 곱해주면 됩니다!! (합성함수의 미분)

&#160; 다음 단계는, 초록색 unit의 weight인 $ w_🍊 $ 에 대한 loss의 변화율인 $ dLoss \over dw_🍊 $ 를 구하는 것입니다. 마찬가지로 합성함수 미분의 원리를 적용하면, 이전 단계에서 구한 $ dLoss \over dw_🍎 $ 에다가 현재 단계의 함수를 미분하여 구할 수 있는 $ dw_🍎 \over dw_🍊 $ 를 곱함으로써 $ dLoss \over dw_🍊 $ 를 계산할 수 있습니다. 이처럼 layer를 역방향으로 거슬러 올라가면서 이 함성함수 미분을 실시하면, 차례대로 모든 unit의 weight에 대한 loss의 변화율들을 구해 나갈 수 있습니다.


&#160;[6단원](#6-forward-propagation과-loss)에서 등장했던 "forward propagation"은 맨 앞 layer부터 연산을 진행했다면, 변화율을 구하는 과정은 마지막 layer부터 진행되기 때문에 "back propagation"이라는 용어를 사용하고 있습니다.

**Gradient Descent (경사하강법) 그리고 Parameter**

&#160;위 내용에서 설명이 이어집니다. 이렇게 각 weight의 gradient를 구했으면, optimization을 하기 위해 기존 weight값에서 gradient를 빼 주면 됩니다. (왜 이런 과정을 거치는지는 위의 "**Optimization 과정**" 단원에서 설명하였습니다.)

<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture17.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진17) Gradient Descent
  </figcaption>
</figure>

&#160;하지만 weight에서 gradient를 그냥 빼지는 않습니다. 우리의 목표는 optimal point에 weight가 수렴하게 만드는 것인데, gradient값이 크면 optimal point를 지나칠 수 있고, 작으면 너무 천천히 다가가게 될 수도 있기 때문입니다. 따라서 우리는 적절한 learning rate ((사진17)에서의 $ \alpha $) 를 gradient에 곱한 후 뺍니다. 

&#160;이처럼 neural network상에 존재하는 다양한 weight들에 대해 gradient에 learning rate를 곱한 값을 빼 줌으로써 업데이트를 해나가는 과정을 gradient descent라고 합니다. 주의해야 할 것은, weight 뿐 아니라 각 unit에 존재하는 bias도 마찬가지로 똑같은 gradient descent 과정이 적용되어 업데이트되어갑니다. 이처럼 학습의 대상이 되는 변수들을 "parameter"이라고 합니다.

&#160;이번 단원에서 배운 내용은 결국 "gradient descent를 수행함으로써 딥러닝 모델을 optimize한다" 라고 정리할 수 있어요.

<br><br>

# 8. 정리하자면..

- 우리의 목표는 neural network가 정확한 예측값을 출력해 내도록 학습 시키는 것이며, '학습'은 neural network상에 존재하는 parameter를 업데이트함으로써 진행됩니다. 
- 데이터셋을 neural network에 집어넣어 예측값을 얻고, ground truth와 예측값을 비교하여 loss를 얻는 과정을 forward-propagation이라고 합니다.
- 각 parameter들에 대한 loss의 변화율인 gradient들을 구한 다음 parameter에 gradient에 learning rate를 곱한 값을 빼 parameter값을 업데이트합니다. 이것을 back-propagation이라고 합니다.
- Forward-propagation, loss 계산, back-propagation을 모두 거치면 1회의 iteration이 일어났다고 할 수 있습니다.
- Iteration 과정이 수천 번, 수만 번, 또는 그 이상으로 반복시켜 학습을 진행시킵니다. 점차 optimization이 진행되어 점점 loss가 작은 예측값을 출력하는 딥러닝 모델이 만들어집니다.

> 이상으로 이번 포스팅을 마치겠습니다. 내용이 길었지만 끝까지 읽어주셔서 감사합니다^^


<br><br><br>

# Reference
[사진1(1)](http://news.heraldcorp.com/view.php?ud=20211103000987) <br>
[사진1(2)](https://www.freecodecamp.org/news/want-to-know-how-deep-learning-works-heres-a-quick-guide-for-everyone-1aedeca88076)<br>
[사진2](https://theory.labster.com/neurons/)<br>
[사진7](https://www.freecodecamp.org/news/want-to-know-how-deep-learning-works-heres-a-quick-guide-for-everyone-1aedeca88076)<br>
[사진8](https://www.datadriveninvestor.com/deep-learning-explained/)<br>
[사진13(1)](https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%B0%A8_%ED%95%A8%EC%88%98)<br>
[사진13(2)](https://www.sfu.ca/~ssurjano/spheref.html)<br>
[사진15](https://vitalflux.com/local-global-maxima-minima-explained-examples/)
