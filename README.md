# Transformer from "PyTorchによる発展DeepLearning"

codes are based on https://github.com/YutaroOgawa/pytorch_advanced

## Usage

### 1. Download data
make "data" directory and download [IMDb_train.tsv](https://drive.google.com/file/d/14N-XRQFaz9jFefg-QLzOteghvphsUyv9/view?usp=sharing) and [IMDb_test.tsv](https://drive.google.com/file/d/1sjygffDlf75iuqsGOUSKZCXlivVOEg-3/view?usp=sharing).

```
data
├── IMDb_test.tsv
└── IMDb_train.tsv
```

### 2. Train
run the command to train the model.

```bash
python -m commands train --data ./data --epochs 10 --batch-size 64 --save-dir weights
```

after training the model, you can find weights in the weights directory.

### 3. Test

run the command to test the model.

```bash
python -m commands test --data ./data --weights ./weights/weights.pth --html-dir ./html
```

after the test, you can find visualized attention weights html files in the html directory.