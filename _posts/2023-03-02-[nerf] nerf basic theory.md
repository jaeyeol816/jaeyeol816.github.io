---
title:  "[NeRF][이론] Neural Radiance Fields for View Synthesis 개념 설명과 논문 리뷰"
categories:
  - neural_representation
use_math: true
---

## 들어가기에 앞서..

- 이 글은 EVCC 2020의 "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" 논문을 바탕으로 작성되었습니다.
- 논문 링크: <https://arxiv.org/abs/2003.08934>
- 이 글은 단순히 논문을 순서대로 해석한 글이 아니라, NeRF를 하나부터 열까지 이해하실 수 있도록 기초 개념부터 시작해서 점점 자세한 내용을 설명하는 식으로 재구성하여 작성되었습니다.

<br><br>

# 목차

순서대로 읽으시는 것을 권장드립니다!

서론
1. Implicit Neural Representation &#160;&#160; [👉바로가기](#1-implicit-neural-representation)
2. NeRF의 목적과 활용
3. 3D particle과 Volume Rendering

큰 그림
4. NeRF Neural Network의 Input과 Output
5. NeRF의 Training및 Infering 과정 요약

깊숙한 내용
6. Hierarchical Volume Sampling
7. Positional Encoding
8. NeRF Neural Network Structure
9. NeRF Volume Rendering
10. Loss Computation

마무리
11. 기존 모델과 비교
12. 결론 및 개선점


<br><br>

# 서론

## 1. Implicit Neural Representation

<figure style="display:block; text-align:center;">
  <img src="/assets/images/nr1/Picture1.jpg"
        style="width: 300px; height: 447px;"> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진1) 모나리자
  </figcaption>
</figure> 

&#160;레오나르도 다 빈치의 대표작 '모나리자'입니다. 1503년도에 레오나르도 다 빈치는 이 그림을 목제 패널에 물감으로 그렸습니다. 이 그림 원본은 목제 패널 위에 유채(油彩)로 '표현(represent)' 되었다고 볼 수 있죠.

&#160;하지만 우리는 지금 이 그림을 루브르 박물관에서 보고 있는 것이 아니라 컴퓨터 화면 속에서 보고 있습니다. 디지털 상에서 '그림'이라는 정보는 어떻게 표현되었다고 볼 수 있을까요? '픽셀(pixel)'로써 표현되었습니다. 이 웹 페이지에 업로드 된 모나리자 그림은 300x447개의 픽셀로 표현된 상태입니다. 각 픽셀은 R,G,B정보를 갖고 있죠.

<figure style="display:block; text-align:center;">
  <img src="/assets/images/nr1/Picture2.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진2) Explicit Representation
  </figcaption>
</figure>

&#160;디지털 상에서 물체는 pixel, vector, voxel, point cloud, mesh 등 다양한 방식으로 표현될 수 있습니다. 이러한 표현 방식은 'Explicit(분명한)' 하다고 할 수 있습니다. 왜냐하면 특정 위치의 색상 정보를 어떤 픽셀이, 또는 어떤 voxel이 가지고 있는지 분명하게 알 수 있기 때문이죠.

<figure style="display:block; text-align:center;">
  <img src="/assets/images/nr1/Picture3.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진3) Implicit Representation
  </figcaption>
</figure>

&#160;이러한 전통적인 representation 방식을 뛰어넘어 neural network를 통해서도 정보를 저장할 수 있습니다. 이 neural network는 특정 좌표를 input으로 넣으면 해당 좌표에 대한 RGB컬러 등의 정보를 output으로 출력해 주는 neural network입니다. 이 경우 Neural network의 파라미터들이 특정 이미지나 물체나 동영상 등의 정보를 담고 있으며, 정보가 neural network의 파라미터들로써 '표현(represent)'된다고 볼 수 있습니다.

&#160;이렇게 neural network를 통해 정보를 표현하는 것을 'implicit(암시된) representation'이라고 합니다. Implicit representation은 각 neuron의 파라미터(weight, bias)들로써 데이터를 저장하기 때문에, 특정 부분의 rgb값이 어디에 담겼는지 그리고 각각의 파라미터가 어떤 의미를 갖고 있는지 인간이 해석할 수 없기 때문에 implicit(암시된)이라는 용어가 붙었다고 보시면 됩니다. 

&#160;그렇다면 implicit neural representation이 기존 방식에 비해 갖는 장점이 무엇일까요? 첫 번째로 저장 공간상의 이점이 있습니다. 3차원 데이터의 경우 voxel 이나 point cloud로써 저장하려면, 3차원상의 모든 점마다 색상값 또는 투명도값을 담아야 하기 때문에 무척 많은 저장 공간을 요구합니다. 하지만 neural network를 통해 저장하면, neural network의 각 unit에 존재하는 파라미터값이 모든 데이터를 함축적으로 담고 있기 때문에 상대적으로 저장 공간을 줄일 수 있습니다. <br>
&#160;두 번째로, 저장 공간이 컨텐츠와 상관없이 일관적이며, 예측 가능합니다. Neural network를 training하기 전 사전에 몇 개의 layer를 사용하고 layer당 몇 개의 unit을 사용할 지 정하면, 메모리를 딱 그만큼만 사용하게 됩니다. 3D 모델을 저장하던, 2D 이미지를 저장하던 상관없이 말이죠. <br>
&#160;하지만, 기존 방식에 비해 데이터를 읽어오는 속도가 느리다는 단점이 있습니다. 매 좌표마다 neural network의 forward연산을 수행하여야만 output값(color등)을 얻을 수 있기 때문이죠.


&#160;NeRF는 하나의 정적인 scene에 대해 implicit neural representation을 적용한 것입니다. 어떤 원리가 숨어 있는지 지금부터 같이 알아보도록 해요~



## Reference
[사진1](https://en.wikipedia.org/wiki/Mona_Lisa) <br>
[사진2(1)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.ripemedia.com%2Fwhy-i-love-pixels-and-so-can-you%2F&psig=AOvVaw3tSX4600BrO4udfnrq6Cb5&ust=1678069785908000&source=images&cd=vfe&ved=0CBEQjhxqFwoTCPCg5Z3fw_0CFQAAAAAdAAAAABAT) <br>
[사진2(2)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fdepositphotos.com%2Fstock-photos%2Fhouse-made-by-voxels.html&psig=AOvVaw0C3X5_XwUuZosPo935QErJ&ust=1678069944652000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCNjxuenfw_0CFQAAAAAdAAAAABAL) <br>
[사진2(3)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mdpi.com%2F2079-9292%2F8%2F10%2F1196&psig=AOvVaw233Dr1LLY9R3Izot94AVN0&ust=1678070060769000&source=images&cd=vfe&ved=0CBEQjhxqFwoTCKDq36Dgw_0CFQAAAAAdAAAAABAf) <br>
