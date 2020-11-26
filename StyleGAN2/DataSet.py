from io import BytesIO

import lmdb
from torch.utils.data import Dataset
from PIL import Image


class MultiResDataset(Dataset):
    def __init__(self, data_path, data_transform, res=256):
        self.env = lmdb.open(
            path=data_path,
            max_readers=64,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset in", data_path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = res
        self.transform = data_transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(idx).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
