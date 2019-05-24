# colorization_pix2pix

* pix2pix 모델을 사용해서 image colorization을 구현한다.
* color space, Loss ftn의 변화하면서 최적 모델을 찾는다.

* [pix2pix](https://phillipi.github.io/pix2pix/) 논문을 참고하여 작성.
***
* 가장 기본적인 모델 구현(LAB color space + cGAN + L1)

| epoch | bw | output | groud-truth|
|:---:|:---:|:---:|:---:|
| 1 | ![e1b](./images/e1b.png) | ![e1o](./images/e1o.png) | ![e1t](./images/e1t.png) |
| 2 | ![e2b](./images/e2b.png) | ![e2o](./images/e2o.png) | ![e2t](./images/e2t.png) |
| 5 | ![e5b](./images/e5b.png) | ![e5o](./images/e5o.png) | ![e5t](./images/e5t.png) |
| 10 | ![e10b](./images/e10b.png) | ![e10o](./images/e10o.png) | ![e10t](./images/e10t.png) |

* 외외로 잘 된다. 

***

## color space 에 따른 학습 경향

| epoch | BW/out/ground-truth |
|:---:|:---:|
| 1 | ![rgb1](./images/rgb1.png) |
| 2 | ![rgb2](./images/rgb2.png) |
| 3 | ![rgb3](./images/rgb3.png) |

* lab color space보다 약간 채도가 낮은 것을 볼 수 있다. RGB의 경우 채널이 3개이기에 학습해야 하는 양이 많기에 속도가 느린것 같다. LAB의 경우 A채널과 B채널만 학습하면 된다.

***

## Loss ftn 구성에 따른 학습.

* cGAN loss only

| epoch | bw | output | groud-truth|
|:---:|:---:|:---:|:---:|
| 1 | ![c1b](./images/c1b.png) | ![c1o](./images/c1o.png) | ![c1t](./images/c1t.png) |
| 2 | ![c2b](./images/c2b.png) | ![c2o](./images/c2o.png) | ![c2t](./images/c2t.png) |
| 5 | ![c5b](./images/c5b.png) | ![c5o](./images/c5o.png) | ![c5t](./images/c5t.png) |
| 10 | ![c10b](./images/c5b.png) | ![c10o](./images/c10o.png) | ![c10t](./images/c10t.png) |

* L1 loss only

| epoch | bw | output | groud-truth|
|:---:|:---:|:---:|:---:|
| 1 | ![l1b](./images/l1b.png) | ![l1o](./images/l1o.png) | ![l1t](./images/l1t.png) |
| 2 | ![l2b](./images/l2b.png) | ![l2o](./images/l2o.png) | ![l2t](./images/l2t.png) |
| 5 | ![l5b](./images/l5b.png) | ![l5o](./images/l5o.png) | ![l5t](./images/l5t.png) |
| 10 | ![l10b](./images/l10b.png) | ![l10o](./images/l10o.png) | ![l10t](./images/l10t.png) |

* cGAN은 타겟이 아니라도 특정한 색을 내는데에 큰 영향을 준다.(색의 다양성 및 채도가 높은 이미지)
* L1항은 Ground truth의 평균적인 색을 따라가기에, 평균적인 회색조, 채도가 낮은 이미지가 생성되었다.
* 두 Loss함수를 모두 사용하게 되면 두 함수의 특성을 모두 얻어 가장 좋은 결과를 얻을 수 있었다.