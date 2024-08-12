import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
from utils.utils import denormalize

def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def augment_image(image):
    """Apply a random augmentation to the image."""
    possible_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ]
    # Randomly choose one transformation to apply
    chosen_transform = random.choice(possible_transforms)
    
    # Apply the chosen transformation
    transformed_image = chosen_transform(image)
    
    return transformed_image

def get_transforms(augmentation=True):
    """Return a set of transformations for training or testing."""
    if augmentation:
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.Resize((224, 224)), #so that it will also has some original images
        ]

        def apply_random_transforms(img):
            # Randomly apply a subset of transformations
            applied_transforms = []
            for aug in augmentations:
                if random.random() < 0.4:  # 40% chance to apply each transform
                    applied_transforms.append(aug)
            applied_transforms.extend([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            transform = transforms.Compose(applied_transforms)
            return transform(img)
        
        return apply_random_transforms
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def balance_dataset_with_augmentation(dataset, target_dir, majority_aug_ratio=0.2):
    """
    Augment classes to balance the dataset while maintaining diversity.
    
    Args:
    - dataset: The original dataset.
    - target_dir: Directory to save augmented images.
    - majority_aug_ratio: Ratio of majority class samples to augment (e.g., 0.2 for 20%).
    
    Returns:
    - augmented_images: List of augmented images.
    - augmented_labels: Corresponding labels for the augmented images.
    """
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(targets)
    max_count = max(class_counts.values())

    augmented_images = []
    augmented_labels = []

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    j = 0
    for class_id, count in class_counts.items():
        augment_count = max_count - count if count < max_count else int(count * majority_aug_ratio)
        class_indices = [i for i, t in enumerate(targets) if t == class_id]
        
        for i in range(augment_count):
            img_idx = random.choice(class_indices)
            data = dataset[img_idx]
            image, label = data[0], data[1]
            image = denormalize(image)
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            
            augmented_image = augment_image(image)
            augmented_image.save(f"{target_dir}/aug_class_{class_id}_{j}.png")

            augmented_images.append(transforms.ToTensor()(augmented_image))
            augmented_labels.append(class_id)
            j += 1

    print("Augmentation complete. Dataset balanced and diversified.")
    return augmented_images, augmented_labels
