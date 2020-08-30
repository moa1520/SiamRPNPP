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

```bash
lr = penalty[best_idx] * score[best_idx] * 0.4
->
lr = penalty[best_idx] * score[best_idx] * 0.01
```
