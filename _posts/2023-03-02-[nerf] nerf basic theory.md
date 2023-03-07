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
2. NeRF 개요 &#160;&#160; [👉바로가기](#2-nerf-개요)

얕게 이해해보기
3. 3D particle과 Ray
4. NeRF의 Training및 Infering 과정 요약

깊게 파헤치기
5. Hierarchical Volume Sampling
6. Positional Encoding
7. NeRF Neural Network Structure
8. NeRF Volume Rendering
9. Loss Computation

마무리
10. 기존 모델과 비교
11. 결론 및 개선점


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


&#160;NeRF(Neural Radiance Field)는 하나의 정적인 scene에 대해 implicit neural representation을 적용한 것입니다. NeRF가 무엇이고 어떤 원리가 숨어 있는지 지금부터 같이 알아보도록 해요.

<br><br>

## 2. NeRF 개요

**NeRF란 무엇일까요?**

&#160;NeRF는 3D 공간 내 입자들의 color값과 density값을 neural network의 파라미터로 표현하는 것입니다. 1단원에서 언급한 용어를 빌리자면, 3D 데이터를 explicit하게 voxel 또는 point cloud으로 나타내는 것이 아니라, 딥러닝을 통해 학습된 neural network 상에서 (implicit하게) 3D 데이터를 reconstruct하는 것이라고 볼 수 있죠. 

<figure style="display:block; text-align:center;">
  <img src="/assets/images/nr1/Picture4.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진4) Input image를 통해 학습한 후, 새로운 view를 합성 가능
  </figcaption>
</figure>

&#160;NeRF모델의 학습은 물체에 대해 다양한 각도에서 촬영된 사진을 가지고 일어납니다. 학습된 NeRF모델이 있다면 우리가 물체를 특정 시점에서 바라봤을 때 어떤 색상값이 보이는지 query할 수 있습니다. 쉽게 말해, 원하는 방향으로 물체를 바라봤을 때의 이미지를 render할 수 있다는 것이죠.

&#160;결국 다양한 각도에서 촬영된 이미지를 갖고 NeRF를 학습시킨 후, 중간 각도에서 바래본 새로운 이미지를 만들어낼 수 있는 것이죠. <br>
&#160;자세히 말하면, NeRF모델을 학습시키기 위해 우리가 필요로 하는 것은 (1) 특정 장면(scene)에 대해 같은 시간 다양한 각도에서 촬영한 2D 이미지들과 (2) 카메라 파라미터(카메라의 위치와 방향 및 초점거리 등의 정보) 입니다. 학습된 NeRF를 통해 알아낼 수 있는 정보는 (1) 3차원 장면 내 특정 입자의 color와 density , 그리고, 직선상의 입자들을 종합하여 계산해낼 수 있는 (2) 특정 viewing direction에서 render된 pixel의 색상 입니다.
<br><br>

**NeRF neural network의 input과 output**

&#160;NeRF 딥레닝 네트워크의 input과 output을 알아봅시다. 정보를 알고싶은 한 3차원 좌표(3d coordinate)와 바라보는 방향(viewing direction)을 집어넣으면 해당 입자의 color와 density값을 얻을 수 있습니다. 

<figure style="display:block; text-align:center;">
  <img src="/assets/images/nr1/Picture5.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진5) 특정 point가 viewing direction에 따라 다른 색상을 가질 수 있음
  </figcaption>
</figure>

&#160;여기서 주목할 부분이 있습니다. NeRF에서는 같은 위치의 입자라도 **바라보는 방향**에 따라 다른 색상값을 출력한다는 것입니다! (사진5)는 같은 장면을 View1와 View2에서 바라본 모습을 각각 나타냅니다. 같은 위치의 입자더라도 View1과 View2에서 빛의 반사에 따라 다른 색상을 가집니다. NeRF에서는 viewing direction도 입력값으로 받음으로써 이러한 성질을 반영하기 때문에 더 정확한 이미지를 출력할 수 있습니다.
<br><br>

**NeRF의 활용**

NeRF가 실제로 어떤 분야에서 어떻게 활용될 수 있을까요?

<figure style="display:block; text-align:center;">
  <img src="/assets/images/nr1/Picture6.png"
        style=""> 
  <figcaption style="text-align:center; font-size:13px; color:#808080">
    (사진6) NeRF의 활용 가능성
  </figcaption>
</figure>

◾️ **가상현실(VR)과 증강현실(AR)**:  NeRF를 통해 실사 기반의 고품질 3D 컨텐츠를 만들어낼 수 있습니다.

◾️ **영화 및 게임 산업**:  현실 세계에 대한 3D 모델을 생성하기 때문에, 영화 및 게임 컨텐츠의 배경을 만드는데 도움이 될 수 있어요.

◾️ **실내 인테리어**:  특정 공간을 NeRF모델로서 표현함으로써, 가구 배치나 실내 디자인을 설계하는데 도움이 될 수 있고, 부동산 거래시에도 참고될 수 있습니다.

◾️ **의학**:  내시경 혹은 CT, MRI등을 통해 촬영된 이미지를 사용해 NeRF를 학습시킴으로써, 신체 내부의 모습에 대해 3D 모델링 할 수 있습니다. 수술 또는 의료 교육 등에 활용할 수 있어요.

◾️ **자율주행 자동차 또는 로보틱스**:  NeRF를 통해 현실 세계에 대한 3D 모델을 생성하여 자율주행 시스템에게 제공함으로써, 더 정확하고 효율적인 경로 선택에 도움을 줄 수 있습니다.

이처럼 NeRF는 다양한 곳에서 응용될 수 있습니다. 이제 NeRF기술을 이해해보고 싶은 마음이 충분히 생기셨을 것 같으니, 다음 단원부터 본격적으로 원리 설명을 시작하겠습니다.



## Reference
[사진1](https://en.wikipedia.org/wiki/Mona_Lisa) <br>
[사진2(1)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.ripemedia.com%2Fwhy-i-love-pixels-and-so-can-you%2F&psig=AOvVaw3tSX4600BrO4udfnrq6Cb5&ust=1678069785908000&source=images&cd=vfe&ved=0CBEQjhxqFwoTCPCg5Z3fw_0CFQAAAAAdAAAAABAT) 
[사진2(2)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fdepositphotos.com%2Fstock-photos%2Fhouse-made-by-voxels.html&psig=AOvVaw0C3X5_XwUuZosPo935QErJ&ust=1678069944652000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCNjxuenfw_0CFQAAAAAdAAAAABAL) 
[사진2(3)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mdpi.com%2F2079-9292%2F8%2F10%2F1196&psig=AOvVaw233Dr1LLY9R3Izot94AVN0&ust=1678070060769000&source=images&cd=vfe&ved=0CBEQjhxqFwoTCKDq36Dgw_0CFQAAAAAdAAAAABAf) <br>
[사진4](https://arxiv.org/abs/2003.08934) <br>
[사진5](https://arxiv.org/abs/2003.08934) <br>
[사진6(1)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.techlearning.com%2Ffeatures%2Fwhat-is-virtual-reality&psig=AOvVaw2zcr4RVRzmst2j1VJZw9PJ&ust=1678247646885000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCPDYwuj1yP0CFQAAAAAdAAAAABAE)
[사진6(2)](https://www.google.com/url?sa=i&url=https%3A%2F%2Ffree3d.com%2F3d-model%2Fhouse-interior--81890.html&psig=AOvVaw1c2zgIz_JEd7g8bCjCb41i&ust=1678247739118000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCJjg0JT2yP0CFQAAAAAdAAAAABAE)
[사진6(3)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.auntminnie.com%2Findex.aspx%3Fsec%3Dlog%26itemID%3D131844&psig=AOvVaw018bA1EkkPFpd5GXi7fS8B&ust=1678248014509000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCJDQ4Jf3yP0CFQAAAAAdAAAAABAE)
[사진6(4)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fpii%2FS2666691X22000136&psig=AOvVaw3suMsxwf7UhS4oApca8_Y-&ust=1678248121256000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCMDv3sr3yP0CFQAAAAAdAAAAABAE)
