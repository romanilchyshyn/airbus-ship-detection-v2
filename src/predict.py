import argparse

import torch
import pandas as pd

from model import load_model
from data import test_loader
from imagenet import normalize_images
from utils import get_device
from rle import mask_to_rle_list

def main():
    args = parse_args()

    device = get_device()

    m = load_model(args.checkpoint_path, device)

    l = test_loader(args.data_dir, batch_size=12)

    rows = []

    with torch.no_grad():
        for images, names in l:
            print(f"predicting images: {names}")
            images = normalize_images(images, device)
            preds = m(images)["out"].argmax(dim=1)

            for p, n in zip(preds, names):
                rle_list = mask_to_rle_list(p.cpu())
                for rle in rle_list:
                    datum = {
                        "ImageId": n,
                        "EncodedPixels": rle
                    }
                    rows.append(datum)

                if not rle_list:
                    datum = {
                        "ImageId": n,
                        "EncodedPixels": ""
                    }
                    rows.append(datum)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_path, index=False)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint-path", type=str)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-path", type=str)

    return p.parse_args()

if __name__ == "__main__":
    main()
