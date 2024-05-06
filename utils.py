import nibabel as nib
import os
import cv2
import numpy as np
import random
from torchvision import transforms
import time
import albumentations as albu
import torch
import skimage.util as skutil

def load_nii(file_path):
    assert os.path.exists(file_path), f"Path {file_path} does not exist"
    img = nib.load(file_path)
    data = img.get_fdata()
    return data


class RandomTransform:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x):
        num_transforms = random.randint(1, len(self.transformations))
        random_transforms = random.sample(self.transformations, num_transforms)
        for transform in random_transforms:
            x = transform(x)
        return x

class DecreaseResolution:
    def __init__(self, iters = 3, size = 128, depth = 3):
        self.iters = iters
        self.size = size
        self.depth = depth
    def __call__(self, img):
        for _ in range(self.depth):
            for _ in range(self.iters):
                img = transforms.Resize((img.shape[2] - self.size, img.shape[3] - self.size), antialias=True)(img)

            for _ in range(self.iters):
                img = transforms.Resize((img.shape[2] + self.size, img.shape[3] + self.size), antialias=True)(img)

        return img