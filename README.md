# kaggle task: Airbus Ship Detection (v2)

[Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)

---

## Train

In project root:
```sh
python3 src/train.py --sample=20000 --epochs=40 --batch-size=18
```

```sh
tensorboard --logdir=runs --bind_all
```

## Predict

```sh
python3 src/predict.py --checkpoint-path="checkpoints/best_model-n.pth" --output-path="test.csv"
```

## Dev

Update `requirements.txt` with:

```sh
pip list --not-required --format=freeze > requirements.txt
```

### Download `data`

kaggle client is sluggish, use curl with link from site.

```sh
export DATA_URL=<url>
curl -o data.zip $DATA_URL
unzip data.zip -d data
```

---
