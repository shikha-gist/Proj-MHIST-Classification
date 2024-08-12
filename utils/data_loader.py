import os
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, WeightedRandomSampler, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter

def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dirs, transform=None, return_no_norm=False):
        """
        Custom dataset for loading images and their corresponding labels.
        
        Args:
        - dataframe (pd.DataFrame): DataFrame containing image names and labels.
        - img_dirs (list or str): Directory or list of directories where images are stored.
        - transform (callable, optional): A function/transform to apply to the images.
        - return_no_norm (bool, optional): Whether to return a non-normalized version of the image.
        """
        self.dataframe = dataframe
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.transform = transform
        self.return_no_norm = return_no_norm
        self.transform_no_norm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image name from DataFrame
        img_name = self.dataframe.iloc[idx, self.dataframe.columns.get_loc('Image Name')]
        
        # Search for the image in all provided directories
        img_path = None
        for dir in self.img_dirs:
            potential_path = os.path.join(dir, img_name)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in any provided directories.")

        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        label = int(self.dataframe.iloc[idx, self.dataframe.columns.get_loc('Majority Vote Label')])

        # Apply transformations
        if self.return_no_norm:
            image_no_norm = self.transform_no_norm(image)
            if self.transform:
                image = self.transform(image)
            return image, image_no_norm, label, img_name
        else:
            if self.transform:
                image = self.transform(image)
            return image, label, img_name

def create_weighted_sampler(dataset):
    """
    Create a WeightedRandomSampler to handle class imbalance.
    
    Args:
    - dataset (Dataset): The dataset to sample from.
    
    Returns:
    - sampler (WeightedRandomSampler): A sampler that samples with weights inversely proportional to class frequency.
    """
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_sample_counts = Counter(targets)
    
    # Calculate weights: inverse of class frequency
    weights = 1. / torch.tensor([class_sample_counts[t] for t in targets], dtype=torch.float)
    
    # Create the sampler
    sampler = WeightedRandomSampler(weights, len(weights))
    
    return sampler

def create_balanced_sampler(dataset):
    """
    Create a SubsetRandomSampler to balance the dataset by oversampling underrepresented classes.
    
    Args:
    - dataset (Dataset): The dataset to sample from.
    
    Returns:
    - sampler (SubsetRandomSampler): A sampler that balances the dataset by oversampling.
    """
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_sample_counts = Counter(targets)
    max_count = max(class_sample_counts.values())
    indices = []

    # Oversample underrepresented classes
    for class_id, count in class_sample_counts.items():
        indices.extend([i for i, t in enumerate(targets) if t == class_id] * (max_count // count))

    return SubsetRandomSampler(indices)
