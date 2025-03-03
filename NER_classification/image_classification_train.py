import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

def get_transforms():
    """Створюємо перетворення для зображень"""
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def split_dataset(data_dir, output_dir, test_size=0.2):
    """Розділяє датасет на тренувальний та тестовий набори."""
    classes = os.listdir(data_dir)
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        images = os.listdir(class_dir)
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
        
        train_class_dir = os.path.join(output_dir, 'train', class_name)
        test_class_dir = os.path.join(output_dir, 'test', class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        for image in train_images:
            os.rename(os.path.join(class_dir, image), os.path.join(train_class_dir, image))
        for image in test_images:
            os.rename(os.path.join(class_dir, image), os.path.join(test_class_dir, image))

def main(args):
    # Замість розділення датасету, використовуємо підготовлений CIFAR-100
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Завантажуємо CIFAR-100
    dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    
    # Фільтруємо тільки класи тварин
    animal_classes = ["bear", "beaver", "chimpanzee", "fox", "kangaroo", "lion", "otter", "porcupine", "raccoon", "wolf"]
    class_to_idx = {cls: dataset.class_to_idx[cls] for cls in animal_classes}
    
    # Створюємо підмножину відфільтрованих даних
    filtered_indices = [i for i, label in enumerate(dataset.targets) if label in class_to_idx.values()]
    
    # Розділяємо на тренувальний і валідаційний набори
    train_size = int(0.8 * len(filtered_indices))
    val_size = len(filtered_indices) - train_size
    train_indices = filtered_indices[:train_size]
    val_indices = filtered_indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Створюємо DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    # Отримуємо імена класів
    class_names = [animal_class for animal_class in animal_classes]