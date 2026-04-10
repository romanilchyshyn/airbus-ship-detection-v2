import os

from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image

import pandas as pd

from .rle import rle_to_mask

class AirbusShipDetectionDataset(Dataset):
    def __init__(self, masks_file, img_dir):
        self.masks_file = masks_file
        self.img_dir = img_dir
        
        self.df = pd.read_csv(masks_file)
        self.df['EncodedPixels'] = self.df['EncodedPixels'].fillna('')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_file = self.df['ImageId'][idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = decode_image(img_path)

        rle = self.df['EncodedPixels'][idx]
        _, h, w = image.shape
        mask = rle_to_mask(rle, h, w)

        return image, mask

def loader(root: str, batch_size=1, shuffle=False, num_workers=0) -> DataLoader:
    mask_path = os.path.join(root, 'train_ship_segmentations_v2.csv')
    img_path = os.path.join(root, 'train_v2')

    dataset = AirbusShipDetectionDataset(masks_file=mask_path, img_dir=img_path)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader
