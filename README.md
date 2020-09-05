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
