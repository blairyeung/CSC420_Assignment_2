import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
import DogBreedDataset
from Config import Config

def rename(name):
    return ' '.join(' '.join(name.split('-')[1:]).split('_'))


def get_training_set():
    dataset = ImageFolder('../SDDsubset')
    breeds = []
    for n in dataset.classes:
        breeds.append(rename(n))

    test_pct = 0.3
    test_size = int(len(dataset) * test_pct)
    dataset_size = len(dataset) - test_size

    val_pct = 0.1
    val_size = int(dataset_size * val_pct)
    train_size = dataset_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    batch_size = Config.BATCH_SIZE

    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        #    transforms.Resize((224, 224)),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        #    transforms.Normalize(*imagenet_stats, inplace=True)

    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #    transforms.Normalize(*imagenet_stats, inplace=True)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #    transforms.Normalize(*imagenet_stats, inplace=True)
    ])

    train_dataset = DogBreedDataset(train_ds, train_transform)
    val_dataset = DogBreedDataset(val_ds, val_transform)
    test_dataset = DogBreedDataset(test_ds, test_transform)

    return train_dataset, val_dataset, test_dataset

    # Create DataLoaders

    def load_data():
        train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_dataset, batch_size * 2, num_workers=2, pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size * 2, num_workers=2, pin_memory=True)


if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = get_training_set()
    img, label = train_dataset[6]
    print(label)
    plt.imshow(img.permute(1, 2, 0))