import os

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import decode_image

import pandas as pd

from rle import rle_list_to_mask

class AirbusShipDetectionDataset(Dataset):
    def __init__(self, masks_file, img_dir, sample: int|None = None, ship_class_weight: float = 0.7):
        self.masks_file = masks_file
        self.img_dir = img_dir
        
        self.df = pd.read_csv(masks_file)
        self.df['EncodedPixels'] = self.df['EncodedPixels'].fillna('')
        
        self.df = self.df.groupby('ImageId')['EncodedPixels'].agg(list).reset_index()

        if sample: 
            empty = self.df[self.df['EncodedPixels'].apply(lambda x: all(p == '' for p in x))]
            non_empty = self.df[self.df['EncodedPixels'].apply(lambda x: any(p != '' for p in x))]
            
            n_empty = int(sample * (1.0 - ship_class_weight))
            n_non_empty = sample - n_empty

            empty_sample = empty.sample(n=n_empty)
            non_empty_sample = non_empty.sample(n=n_non_empty)

            self.df = pd.concat([empty_sample, non_empty_sample]).sample(frac=1).reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_file = self.df['ImageId'][idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = decode_image(img_path)

        rles = self.df['EncodedPixels'][idx]
        _, h, w = image.shape
        mask = rle_list_to_mask(rles, h, w)

        return image, mask

def dataset(root: str, sample: int|None = None):
    ds = AirbusShipDetectionDataset(
        img_dir=os.path.join(root, 'train_v2'),
        masks_file=os.path.join(root, 'train_ship_segmentations_v2.csv'),
        sample=sample,
    )

    return ds

def loader(
        root: str, 
        sample: int|None = None, 
        batch_size=1, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    ) -> DataLoader:
    l = DataLoader(
        dataset(root, sample=sample),
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    return l

def train_val_loader(
        root: str, 
        val_split: float, 
        sample: int|None = None, 
        batch_size=1, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    ) -> DataLoader:
    ds = dataset(root, sample=sample)

    train_ds, val_ds = random_split(ds, [1-val_split, val_split])

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    return train_loader, val_loader

class AirbusShipDetectionDatasetTest(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = decode_image(img_path)

        return image, img_name

def dataset_test(root: str):
    ds = AirbusShipDetectionDatasetTest(img_dir=os.path.join(root, 'test_v2'))

    return ds

def test_loader(
        root: str, 
        batch_size=1, 
        num_workers=4, 
    ) -> DataLoader:
    ds = dataset_test(root)

    test_loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
    )

    return test_loader
