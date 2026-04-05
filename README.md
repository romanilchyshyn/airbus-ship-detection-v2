# kaggle task: Airbus Ship Detection (v2)

[Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)

---

## Dev

Update `requirements.txt` with:

```sh
pip list --not-required --format=freeze > requirements.txt
```

### Download `data`

kaggle client is sluggish, use curl with link form site.

```sh
export DATA_URL=<url>
curl -o data.zip $DATA_URL

```
