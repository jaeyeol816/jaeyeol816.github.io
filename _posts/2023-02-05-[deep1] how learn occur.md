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

## 목차

순서대로 읽으시는 것을 추천드립니다!
1. '인공 신경망'과 '신경망' &#160;&#160; [👉바로가기](#1-인공-신경망과-사람의-신경망)
2. 퍼셉트론 (Weight & Activation)  &#160;&#160; [👉바로가기](#2-퍼셉트론-weight--activation)
3. 다중 퍼셉트론  &#160;&#160; [👉바로가기](#3-다중-퍼셉트론)
4. Layer  &#160;&#160; [👉바로가기](#4-unit과-layer)
5. 학습이란? Weight를 찾아가는 과정이다! [👉바로가기](#5-학습이란-weight를-조정해-가는-과정이다)
6. 투과 시키기 (Forward-propagation)
7. Loss와 Gradient
8. Weight를 조정해 학습 시키기 (Back-propagation) 
9. 정리하자면...


<br>

## 1. '인공 신경망'과 '사람의 신경망'

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

<br>

## 2. 퍼셉트론 (Weight & Activation)

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

<br> 

## 3. 다중 퍼셉트론
시간이 없으신 분들은 이번 절은 건너뛰어도 좋아요.<br>
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


<br>

## 4. Unit과 Layer
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture7.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진7) Unit과 Layer
  </figcaption>
</figure>
**Unit**<br>
&#160;이제 딥러닝에서 쓰이는 용어로 넘어가도록 합시다. 지금까지 '뉴런(neuron)', '퍼셉트론(perceptron)'이라고 불렀던 것을 딥러닝에서는 '노드(node)' 또는 '유닛(unit)'이라고 불러요. 저는 앞으로 유닛(unit)이라는 표현을 사용하도록 하겠습니다.

**Layer**<br>
&#160;이전 단원에서 다중 퍼셉트론을 설명할 때 NAND유닛과 OR유닛이 하나의 '층'을 이루어고, AND유닛이 하나의 '층'을 이루었습니다. 이러한 '층'을 '레이어(layer)'라는 용어로 부릅니다. 
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture8.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진8) Layer의 종류
  </figcaption>
</figure>

**Input Layer, Hidden Layer, Output Layer**<br>
&#160;Layer는 크게 세 종류로 나눌 수 있습니다. '입력층(Input Layer)', '은닉층(Hidden Layer)', '출력층(Output Layer)'으로 말이죠. Input Layer는 네트워크에 들어오는 데이터 그 자체를 지칭합니다. <br>
&#160;네트워크로 특정 사람의 '[눈 크기, 코 높이, 머리 색상]' 과 같은 정보가 들어오면 네트워크가 '호감' 또는 '비호감'을 판단한다고 칩히다. 이때 들어오는 데이터 자체 (예를 들어 [3cm, 1cm, 갈색])가 Input Layer를 의미합니다. 이때 데이터가 3개 원소로 이루어져 있으니 Input Layer의 Unit의 개수는 3개입니다. 최종적으로 '호감' 또는 '비호감'이라는 데이터가 출력되는 Layer을 Output Layer라고 합니다. Output Layer는 마지막 Layer가 되는 거죠. <br>
&#160;Input Layer도 아니고 Output Layer도 아닌 모든 Layer들을 'Hidden Layer'라고 합니다. 중간 Layer의 이름에 Hidden이라는 용어가 들어간 이유은 무엇일까요? Hidden Layer으로 입력되거나 Hidden Layer에서 출력되는 값은 사람의 입장에서 의미있게 해석되는 값이 아니기 때문에 '감춰졌다'라는 표현을 사용합니다. 이 값들이 최종 Output Layer의 결과물을 만드는데 영향은 주지만, 값 하나하나가 의미를 갖고 있지 않습니다.

**Layer간 연결의 특징**<br>
&#160;특정 N번째 Layer가 있다고 칩시다. N번째 Layer의 모든 특정 Unit의 출력은 N+1번째 Layer의 모든 Unit으로 전달된다는 특징이 있습니다. 다르게 말해서, N번째 Layer의 개별 Unit 하나하나는 N-1번째 Layer의 모든 Unit으로부터 Input값을 받는다고 할 수 있죠! (사진7과 사진8을 세심히 관찰해 보시면 무슨 말인지 알 수 있을 거에요.)<br>
&#160; 한 Unit에서는 이전 Layer의 모든 Unit으로부터 정보를 얻어와 계산(Linear Function + Activation) 을 한 후, 그 결과값을 다음 Layer의 모든 Unit으로 전달합니다.

**Layer의 개수** <br>
&#160;일반적으로 Layer의 개수를 셀 때 input layer의 개수는 포함시키지 않아요. 사실 input layer는 연산을 하는 것이 아니라 네트워크에 주어진 데이터 그 자체이기 때문이죠. 따라서 (사진7), (사진8)의 경우 Layer 개수는 5개라고 할 수 있습니다.

<br> 
## 5. 학습이란? Weight를 조정해 가는 과정이다!

&#160;Neural Network를 하나의 "함수"라고 생각해 봅시다. 우리가 이 함수에 데이터를 입력하면, 함수는 예측값을 출력합니다. 이때 예측값을 정확하게, 즉 실제값과 비슷하게 출력하도록 만드는 것이 우리의 목표입니다.
<figure style="display:block; text-align:center;">
  <img src="/assets/images/deep1/Picture9.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진9) 학습이 반복됨에 따라 weight값들이 업데이트됨
  </figcaption>
</figure>
&#160;지금까지 배웠듯이 정해진 구조의 Neural Network의 아웃풋을 결정하는 것은 각 Unit의 weight입니다. 따라서 '학습'이라는 것은 최적의 예측값을 만들도록 각 Unit의 weight들이 업데이트되는 과정입니다. 학습이 진행됨에 따라 weight는 점차 변화합니다. <br>
> &#160;믈론 weight뿐 아니라 각 Unit에 linear function부분에 존재하는 bias도 업데이트 하지만, 이는 다음 포스팅에서 자세히 다뤄 보도록 하겠습니다. (이번 포스팅에서는 weight에 대해서만 언급하겠습니다)

&#160;그렇다면 weight는 어떻게 업데이트 되는 것일까요? 매 반복 주기(Iteration)마다 input데이터에 대한 예측값이 출력되고, 예측값에 대한 오차(loss)를 계산한 후, 현재 weight들에 대한 loss의 "변화율"에 기반하여 weight들을 업데이트해나갑니다. 이러한 반복이 진행되면 진행될수록 weight들은 더 낮은 loss를 갖는 예측값을 출력하도록 변화하게 됩니다. 어렵게 느껴지시나요? 다음 절에서 설명이 이어집니다.

&#160;Deep Learning이라는 용어의 뜻도 어렵게 이해할 필요가 없어요. "Deep" 하다는 것은 layer가 여러 겹 있다는 뜻이고, "Learning" 은 그 상태에서 각 layer의 weight를 업데이트해가며 학습을 한다는 것입니다!

<br>

## 6. 투과시키기 (Forward Propagation)



<br><br><br>

## Reference
[사진1(1)](http://news.heraldcorp.com/view.php?ud=20211103000987) <br>
[사진1(2)](https://www.freecodecamp.org/news/want-to-know-how-deep-learning-works-heres-a-quick-guide-for-everyone-1aedeca88076)<br>
[사진2](https://theory.labster.com/neurons/)<br>
[사진7](https://www.freecodecamp.org/news/want-to-know-how-deep-learning-works-heres-a-quick-guide-for-everyone-1aedeca88076)<br>
[사진8](https://www.datadriveninvestor.com/deep-learning-explained/)<br>
