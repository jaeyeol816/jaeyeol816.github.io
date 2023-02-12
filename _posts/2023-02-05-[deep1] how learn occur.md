---
title:  "[딥러닝1] 딥러닝에서 학습은 어떻게 일어날까?"
categories:
  - deep_learning
use_math: true
---

## 들어가기에 앞서...

> - 이 글은 딥러닝을 처음 접하시는 분들을 위해 개념적이고 추상적인 내용만 담았습니다. 다음 포스팅에서 딥러닝의 수학적인 이해를 다루겠습니다.
> - 일부 용어에 있어서 한국어로 번역된 것보다 영어를 사용하는 것을 지향합니다. 이해해 주시면 감사하겠습니다!
> - 잘못된 내용이 있으면 언제든 피드백 주세요! 빠르게 고치도록 하겠습니다 🥰

<br>

## 목차

순서대로 읽으시는 것을 추천드립니다!
1. '인공 신경망'과 '신경망' &#160;&#160; [👉바로가기](#1-인공-신경망과-사람의-신경망)
2. 퍼셉트론 (Weight & Activation)  &#160;&#160; [👉바로가기](#2-퍼셉트론-weight--activation)
3. 다중 퍼셉트론  &#160;&#160; [👉바로가기](#3-다중-퍼셉트론)
3. Layer
4. 주어지는 데이터 예시
5. 학습이란? Weight를 찾아가는 과정이다!
5. 투과 시키기 (Forward-propagation)
6. Loss와 Gradient
7. Weight를 조정해 학습 시키기 (Back-propagation) 
8. 정리하자면...


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


## 2. 퍼셉트론 (Weight & Activation)

&#160;딥러닝 모델을 흔히 MLP (Multi Layer Perceptron) 이라고 합니다. 퍼셉트론이 여러 Layer로 이루어진 구조라고 해석할 수 있습니다. 딥러닝 모델에서 퍼셉트론은 한 뉴런에서의 연산을 담당합니다. 이에 대응하는 사람의 뉴런(신경세포)에서의 연산을 간단히 알아봅시다.

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

&#160;퍼셉트론은 신경세포의 동작 과정을 통계학적으로 모델링한 알고리즘입니다. (사진3)을 보며 이해해 봅시다. 하나의 퍼셉트론은 여러 Input을 가질 수 있습니다. (사진3)에서 🍎, 🍏, 🍊는 들어오는 Input을 나타냅니다. 이러한 Input들에게 첫번째로 Weighted Sum(가중합)을 구하는 연산이 적용됩니다. 🍎(첫번째 input)에는 $ w_1 $, 🍏(두번째 input)에는 $ w_2 $, 🍊(세번째 input)에는 $ w_3 $ 이 곱해지게 되죠. <br>
&#160;구해진 가중합을 $ Z $라고 합시다. 이후 Activation Function(활성 함수)를 거치게 되는데, $Z$값이 특정 Threshold(임계값) 이상이면 1이 출력되고, 이하이면 0을 출력하게 됩니다. <br>
&#160;실제 딥러닝에서는 Activation Function으로 위와 같은 계단 함수 말고 Sigmoid Function, ReLU Function등이 쓰입니다. 다음 포스팅에서 소개해 드리도록 하죠.

&#160;뉴런으로 들어온 정보가 종합(Linear Function 적용)된 후, 임계점 이상이면(Activation Function 적용) 1을 다음 뉴런으로 넘긴다는 점이 사람의 뉴런과 비슷하게 느껴지시지 않나요? <br>
&#160;퍼셉트론 한 개를 거쳤을 때 일어나는 일은 이해를 하셨을 겁니다. 하나의 퍼셉트론만 보셨을 때는 통계상의 Regression(회귀)와 다를 것이 없어 보입니다. 하지만 퍼셉트론이 무서운 점은 여러 개 모였을 때 만들어낼 수 있는 놀라운 결과에 있습니다. 다음 절을 참고해주세요!


## 3. 다중 퍼셉트론







<br>

## Reference
[사진1(1)](http://news.heraldcorp.com/view.php?ud=20211103000987) <br>
[사진1(2)](https://www.freecodecamp.org/news/want-to-know-how-deep-learning-works-heres-a-quick-guide-for-everyone-1aedeca88076)
[사진2](https://theory.labster.com/neurons/)

