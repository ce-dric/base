#### 기초가 부족 한 것 같아서 base 리포지토리를 만들어 봄

>Anaconda 로 가상환경을 설정,
```console
$ conda create -n [ENV_NAME] python=3.8 -y
$ conda activate [ENV_NAME]
```
> 필요한 라이브러리를 설치, [pytorch](https://pytorch.org/)
```console
$ pip install -r requirements.txt
$ conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```
> Mac (M1)
- MPS acceleration is available on MacOS 12.3+
```console
$ conda install pytorch torchvision -c pytorch
```

---
## CASE
[MVTEC-CAPSULE](docs/mvtec_capsule.md)
