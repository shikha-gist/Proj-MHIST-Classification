import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from models.models import get_model
from models.grad_cam import GradCAM, CustomImageDataset
from utils.utils import denormalize
import glob

def main():
    # Load configuration from the YAML file
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the directory where Grad-CAM images will be saved
    save_dir = os.path.join('gradcam_images', config['data']['model_name'])
    os.makedirs(save_dir, exist_ok=True)

    # Load the model architecture and parameters
    model = get_model(**config['model'])
    model = model.to(device)

    # Load the best model checkpoint
    model_path = os.path.join(config['data']['model_save_dir'], config['data']['model_name'])
    best_model_files = glob.glob(os.path.join(model_path, 'best_model_epoch_*.pth'))

    if best_model_files:
        best_model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = best_model_files[-1]
        print(f"Loading best model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise FileNotFoundError("No best model file found.")

    # Define normalization and denormalization transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the test dataset
    df = pd.read_csv(config['data']['annotations_file'])
    test_df = df[df['Partition'] == 'test']
    test_dataset = CustomImageDataset(dataframe=test_df, img_dir=config['data']['images_dir'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize Grad-CAM based on the model architecture
    if config['model']['architecture'].startswith('resnet'):
        target_layer = getattr(model, 'layer4')[-1]  # Adjust the layer as needed for ResNet
    elif config['model']['architecture'].startswith('vit'):
        target_layer = getattr(model, 'vit').encoder.layer[-1]  # Adjust for ViT
    else:
        raise ValueError(f"Unsupported architecture: {config['model']['architecture']}")

    # Unfreeze the target layer to enable Grad-CAM computation
    for param in target_layer.parameters():
        param.requires_grad = True

    gradcam = GradCAM(model=model, target_layer=target_layer)

    # Generate Grad-CAM heatmaps for 10 sample of images
    for i, (image, label, img_name) in enumerate(test_loader):
        if i >= 10:
            break
        image = image.to(device)
        heatmap = gradcam.generate_heatmap(image)
        image = denormalize(image.squeeze().cpu())
        image = transforms.ToPILImage()(image)
        overlayed_image, resized_heatmap = gradcam.overlay_heatmap(heatmap, image)

        # Create a figure with 3 subplots and save it
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title(f'Original Image\nLabel: {label}')
        axes[0].axis('off')

        heatmap_img = axes[1].imshow(resized_heatmap, cmap='jet', interpolation='nearest')
        fig.colorbar(heatmap_img, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title(f'Heatmap\nImage: {img_name}')
        axes[1].axis('off')

        axes[2].imshow(overlayed_image)
        axes[2].set_title(f'Overlayed Heatmap\nImage: {img_name}')
        axes[2].axis('off')

        # Save the figure
        figure_path = os.path.join(save_dir, f'{os.path.basename(img_name[0])}_gradcam.png')
        fig.savefig(figure_path)
        plt.close(fig)

        print(f"Saved {figure_path}")

if __name__ == "__main__":
    main()
