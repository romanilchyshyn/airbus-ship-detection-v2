# kaggle task: Airbus Ship Detection (v2)

[Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)

---

## Train

In project root:
```sh
python3 src/train.py --sample=1000 --epochs=30 --batch-size=10 --run-name="tb-$(date +"%Y-%m-%dT%H:%M:%S%z")"
```

```sh
tensorboard --logdir=runs --bind_all
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

Train on 4060 performance:
Sampled dataset 5k   items - 10m  per epoch.      * 30 = 300m   ~ 5h
Real dataset    200k items ~ 400m per epoch (7h). * 30 = 12000m ~ 8d
