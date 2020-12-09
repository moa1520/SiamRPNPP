# SiamRPN++ with PyTorch

- Demo Version

- [x] model load
- [x] backbone
- [x] neck
- [x] rpn
- [x] tracker

### Download models

Download models in PySOT Model Zoo and put the model.pth in the correct directory in pretrained_model

### Video directory

Make directory named 'data'

### Test demo

```bash
python demo.py --video_name data/bag.avi
```

### demo.py -> Noaml 2D Image

### demo_focal.py -> Plenoptic Image

### 수정한 부분

#### tracker.py

```bash
lr = penalty[best_idx] * score[best_idx] * 0.4
->
lr = penalty[best_idx] * score[best_idx] * 0.05
```

0.4 대신 0.05로 대체 (BBOX가 과도하게 커지는 것을 막음)

### demo_cls.py

tracker.py에 get_cls 함수 추가해서
전체 focal에 대한 response값들을 한번에 softmax함수에 적용
20~30 범위로는 잘 돌아가나, 더 큰 범위에서는 OOM에러 발생

### 환경
- OS : Windows10
- Python 3.7.9
- Cuda V11.1
- PyTorch 1.7.0

### 데이터 폴더 형태

Plenoptic image 데이터 추적

```bash
- NonVideo4
  - 001
    - images
      - 001.png
      - 002.png
      - 003.png
      ...
    - focals
      - 001.png
      - 002.png
      - 003.png
      ...
  - 002
    - images
    - focals
  - 003
  - ...
  - n
```

### 가상환경 설치

메인 폴더 안에 requirements.txt 포함되어 있음
터미널로 폴더 경로로 들어가 아래 명령어 실행

```bash
pip install -r requirements.txt
```

### Args

- video_name* (str) : 영상, 이미지의 최상위 루트 ex) E:/NonVideo4
- type        (str) : 영상 타입 (2D, 3D) default: 2D
- img2d_ref*   (str) : 객체 추적 시 보여질 메인 이미지 상대 경로 ex) images/005.png
- gt_on      (bool) : GT를 보이게 하며 성능을 측정 할 것인지 True or False, 기본값: False
- record      (bool) : 추적 이미지를 저장할 것인지, 기본값: False
- start_num*    (int) : 탐색할 Focal 이미지 시작 범위 지정, 기본값: 20
- last_num*     (int) : 탐색할 Focal 이미지 끝 범위 지정, 기본값: 50


### start_num, last_num 테스트 기준

- NonVideo4 기준
start_num : 20, last_num : 50 으로 테스트 진행

- Video3 기준
start_num : 70, last_num : 85 으로 테스트 진행


- 일반 2D 이미지 추적 시 -> demo.py 실행

- Plenoptic 이미지 추적 시 -> demo_focal.py 실행


### Plenoptic 빠른 실행 예시

```bash
python demo_focal.py --video_name 경로/NonVideo4 --type 3D --img2d_ref images/005.png
```

경로 부분에 NonVideo4 폴더가 있는 경로를 삽입
