import os
import pickle
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
import tqdm
import re
import torch
import cv2
from torchvision.transforms import GaussianBlur

class ImageLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def __iter__(self):
        dataset = self.dataset
        dataset_size = len(dataset)

        for i in range(0, dataset_size, self.batch_size):
            start_index = i
            end_index = min(i + self.batch_size, dataset_size)

            batch_indices = slice(start_index, end_index)

            batch = dataset[batch_indices]

            yield batch

class ImageDataset(Dataset):
    def __init__(self, path = "./data/", batch_size = 128):
        assert os.path.exists(path), f"Path {path} does not exist"
        self.path = path
        self.files = [x for x in os.listdir(path) if x.endswith(".gz")]
        self.cache = {}
        self.batch_size = batch_size

    def _get_slices(self, indices):
        pickle_dir = os.path.join(self.path, "pkl")
        os.makedirs(pickle_dir, exist_ok=True)
        if len(os.listdir(pickle_dir)) > 0:
            pkls = [x for x in os.listdir(pickle_dir)]
            assert len(pkls) > 0, "No pickles found."
            pattern = r'\((\d+)\)\((\d+)\)'
            pkls = sorted(pkls, key=lambda x: int(re.search(pattern, x).group(2)))
            size = int(re.search(pattern, pkls[0]).group(1))

            data_slices = []

            if isinstance(indices, int):
                indices = slice(indices, indices + 1)

            assert isinstance(indices, slice), "Indices must be a slice or integer"

            start = indices.start or 0
            stop = indices.stop or len(self)
            step = indices.step or 1

            for index in range(start, stop, step):
                fil = index // size
                i = abs(size - ((size * (fil + 1))) + index)
                
                if fil in self.cache:
                    data = self.cache[fil]
                else:
                    path = os.path.join(pickle_dir, pkls[fil])
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    self.cache[fil] = data 
                k = data[i, :, :]
                k = k.astype(np.uint8)
                k = cv2.cvtColor(k, cv2.COLOR_GRAY2RGB)
                k = (k - np.min(k)) / (np.max(k) - np.min(k))
                k = np.where(k > 0.25, k, 0)

                data_slices.append(k)
            return data_slices
        else:
            c = 0
            for file in tqdm.tqdm(self.files):
                m, n = self.batch_size, 0
                path = os.path.join(self.path, file)
                data = load_nii(path)
                print(f"{file}: {data.shape[0]}")
                data = np.apply_along_axis(lambda x: x / 255, 0, data)
                while n < data.shape[0]:
                    part_path = os.path.join(pickle_dir, f"({min(data.shape[0] - n, self.batch_size)})({c}).pkl")
                    partition = data[n:m, :, :]
                    if not os.path.exists(part_path):
                        with open(part_path, 'wb') as f:
                            pickle.dump(partition, f)
                    n = m
                    m += min(data.shape[0] - n, self.batch_size)
                    c += 1
                    self.cache[part_path] = partition
                del data

        return self._get_slices(index)
    
    def __len__(self):
        pkls = [x for x in os.listdir(os.path.join(self.path, "pkl"))]
        assert len(pkls) > 0, "No pickles found."
        pattern = r'\((\d+)\)\((\d+)\)'
        pkls = sorted(pkls, key=lambda x: int(re.search(pattern, x).group(2)))
        total_size = 0
        for pkl in pkls:
            size = int(re.search(pattern, pkl).group(1))
            total_size += size
        return total_size
    
    def __getitem__(self, indices):
        if isinstance(indices, slice):
            start = indices.start if indices.start is not None else 0
            stop = indices.stop if indices.stop is not None else len(self)
            return LazySliceDataset(self, start, stop)
        slices = self._get_slices(indices)
        batch = torch.stack([torch.tensor(slice_data, dtype=torch.float32) for slice_data in slices])
        batch = batch.permute(0, 3, 2, 1)
        batch = torch.nn.functional.interpolate(batch, size=(512, 512), mode='bilinear', align_corners=False)
        return batch
    
class LazySliceDataset(Dataset):
    def __init__(self, dataset, start, stop):
        self.dataset = dataset
        self.start = start
        self.stop = stop

    def __getitem__(self, index):
        start = index.start if index.start is not None else 0
        stop = index.stop if index.stop is not None else len(self)
        return torch.stack([self.dataset[i] for i in range(self.start + start, self.start + stop)]).squeeze(0)

    def __len__(self):
        return self.stop - self.start