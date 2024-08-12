import os
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.models import get_model
from utils.data_loader import CustomImageDataset, create_weighted_sampler, create_balanced_sampler
from utils.transforms import get_transforms, balance_dataset_with_augmentation
from utils.training import train_model, save_top_and_bottom_examples, plot_confusion_matrix, compute_confusion_matrix
from utils.utils import set_seed, read_metrics, denormalize
import torch.optim as optim
import matplotlib.pyplot as plt
import glob

def main():
    # Load configuration from the YAML file
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set random seed for reproducibility across runs
    set_seed(config['random_seed'])

    # Load dataset annotations from a CSV file
    df = pd.read_csv(config['data']['annotations_file'])

    # Convert string labels ('SSA' and 'HP') to integer labels (1 and 0 respectively)
    label_map = {'SSA': 1, 'HP': 0}
    df['Majority Vote Label'] = df['Majority Vote Label'].map(label_map)

    # Split dataset into training and test sets based on the 'Partition' column
    train_df = df[df['Partition'] == 'train']
    test_df = df[df['Partition'] == 'test']

    # Define transformations: one with augmentation (training) and one without (testing)
    transform = get_transforms(augmentation=config['data']['augmentation'])
    transform_no_augmentation = get_transforms(augmentation=False)

    # Create datasets using the custom dataset class
    train_dataset = CustomImageDataset(dataframe=train_df, img_dirs=config['data']['images_dir'], transform=transform)
    test_dataset = CustomImageDataset(dataframe=test_df, img_dirs=config['data']['images_dir'], transform=transform_no_augmentation)

    # Balance the dataset by augmenting underrepresented classes (adding augmented images in training)
    if config['data'].get('balance_dataset', False):
        augmented_images, augmented_labels = balance_dataset_with_augmentation(
            train_dataset, 
            target_dir=config['data']['augmented_data_dir'],  # Save augmented images to the specified directory
        )

        # Create a new DataFrame for the augmented data
        augmented_df = pd.DataFrame({
            'Image Name': [f"aug_class_{label}_{i}.png" for i, label in enumerate(augmented_labels)],
            'Majority Vote Label': augmented_labels,
            'Partition': ['train'] * len(augmented_labels)
        })

        # Combine the original and augmented data
        train_df = pd.concat([train_df, augmented_df], ignore_index=True)

        # Reload the dataset to include both original and augmented data
        train_dataset = CustomImageDataset(dataframe=train_df, img_dirs=[config['data']['images_dir'], config['data']['augmented_data_dir']], transform=transform)

    # Split the training dataset into training and validation sets
    train_size = int(config['data']['train_val_split'] * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create dataloaders for training, validation, and test sets
    if config['data']['class_balancing']:
        train_sampler = create_balanced_sampler(train_dataset) if config['model']['oversample_minority_class'] else create_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=config['model']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['model']['batch_size'], shuffle=False)
    
    # Define the directory where the model will be saved
    model_path = os.path.join(config['data']['model_save_dir'], config['data']['model_name'])
    
    # Create the model directory if it doesn't exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Load the model with the specified architecture and parameters from the config
    model = get_model(**config['model'])

    # Display trainable layers in the model
    print("Trainable layers in the model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} is trainable.")

    # Count the number of trainable and non-trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of non-trainable parameters: {non_trainable_params}")

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the optimizer with parameters from the config
    optimizer = optim.Adam(model.parameters(), lr=config['model']['lr'], weight_decay=config['model']['weight_decay'])

    # Resume training from the last checkpoint if enabled
    checkpoint_path = os.path.join(model_path, 'last_checkpoint.pth')
    start_epoch = 0
    resume_training = config['training']['resume']

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Train the model using the defined training function
    train_model(
        model, 
        train_loader, 
        val_loader, 
        config['model'], 
        model_path, 
        device,
        start_epoch,  # Start from the last checkpoint epoch
        resume=resume_training  # Pass the resume flag to the training function
    )

    # Load the best model from saved checkpoints based on the epoch number
    best_model_pattern = os.path.join(model_path, 'best_model_epoch_*.pth')
    best_model_files = glob.glob(best_model_pattern)

    if best_model_files:
        # Sort to find the best epoch and load it
        best_model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_model_path = best_model_files[-1]  # Load the most recent best model
        best_epoch = int(best_model_path.split('_')[-1].split('.')[0])
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        raise FileNotFoundError("No best model file found.")

    # Define the directories where the metrics files are saved
    train_dir = os.path.join(config['data']['eval_save_dir'], config['data']['model_name'], 'train_examples')
    val_dir = os.path.join(config['data']['eval_save_dir'], config['data']['model_name'], 'val_examples')
    test_dir = os.path.join(config['data']['eval_save_dir'], config['data']['model_name'], 'test_examples')


    # Evaluate the model and save top/bottom/medium examples for train, validation, and test sets
    for dataset_name, dataset_loader, save_dir in [('Train', train_loader, train_dir), ('Validation', val_loader, val_dir), ('Test', test_loader, test_dir)]:
        save_top_and_bottom_examples(
            model, 
            dataset_loader, 
            save_dir, 
            device, 
            best_epoch=best_epoch, 
            s_type=f'{dataset_name} Set:'
        )

    # Compute confusion matrices for each dataset
    train_cm = compute_confusion_matrix(model, train_loader, device)
    val_cm = compute_confusion_matrix(model, val_loader, device)
    test_cm = compute_confusion_matrix(model, test_loader, device)

    # Ensure overall performance directory exists
    overall_performance_dir = os.path.join(config['data']['eval_save_dir'], config['data']['model_name'], 'overall_performance', f"epoch_{best_epoch}")
    if not os.path.exists(overall_performance_dir):
        os.makedirs(overall_performance_dir)

    # Save confusion matrices as images
    for cm, dataset_name in [(train_cm, 'train'), (val_cm, 'val'), (test_cm, 'test')]:
        plot_confusion_matrix(cm, classes=['HP', 'SSA'], title=f'Confusion Matrix - {dataset_name.capitalize()} Data',
                              save_path=os.path.join(overall_performance_dir, f'{dataset_name}_confusion_matrix.png'))

    # Function to read metrics from a text file
    def read_metrics(file_path):
        metrics = {}
        with open(file_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                metrics[key] = float(value)
        return metrics

    # Load metrics for each dataset and extract accuracy and AUC values
    train_metrics = read_metrics(os.path.join(train_dir, f"epoch_{best_epoch}", 'performance_metrics.txt'))
    val_metrics = read_metrics(os.path.join(val_dir, f"epoch_{best_epoch}", 'performance_metrics.txt'))
    test_metrics = read_metrics(os.path.join(test_dir, f"epoch_{best_epoch}", 'performance_metrics.txt'))

    datasets = ['Train', 'Validation', 'Test']
    accuracy_values = [train_metrics['Accuracy'], val_metrics['Accuracy'], test_metrics['Accuracy']]
    auc_values = [train_metrics['AUC'], val_metrics['AUC'], test_metrics['AUC']]

    # Plot and save Accuracy bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(datasets, accuracy_values, color=['blue', 'orange', 'green'])
    plt.title('Accuracy for Train, Validation, and Test Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracy_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    plt.savefig(os.path.join(overall_performance_dir, 'accuracy_bar_chart.png'))
    plt.close()

    # Plot and save AUC bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(datasets, auc_values, color=['blue', 'orange', 'green'])
    plt.title('AUC for Train, Validation, and Test Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('AUC')
    plt.ylim(0, 1)
    for i, v in enumerate(auc_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    plt.savefig(os.path.join(overall_performance_dir, 'auc_bar_chart.png'))
    plt.close()

    print("Training complete and examples saved.")

if __name__ == '__main__':
    main()
