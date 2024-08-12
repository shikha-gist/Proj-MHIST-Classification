import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from models.models import get_model
from utils.data_loader import CustomImageDataset
from utils.transforms import get_transforms
from utils.training import save_top_and_bottom_examples, plot_confusion_matrix, compute_confusion_matrix
from utils.utils import set_seed
import matplotlib.pyplot as plt
import glob

def main():
    # Load configuration from YAML file
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set random seed for reproducibility
    set_seed(config['random_seed'])

    # Load dataset annotations from CSV file
    df = pd.read_csv(config['data']['annotations_file'])

    # Map string labels ('SSA' and 'HP') to integer labels (1 and 0 respectively)
    label_map = {'SSA': 1, 'HP': 0}
    df['Majority Vote Label'] = df['Majority Vote Label'].map(label_map)

    # Filter test dataset based on 'Partition' column
    test_df = df[df['Partition'] == 'test']

    # Define transformations (without augmentation for testing)
    transform = get_transforms(augmentation=False)

    # Create the test dataset
    test_dataset = CustomImageDataset(dataframe=test_df, img_dirs=config['data']['images_dir'], transform=transform)

    # Create the test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config['model']['batch_size'], shuffle=False)

    # Load the model architecture and parameters
    model = get_model(**config['model'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the directory paths for saving models and evaluation results
    model_path = os.path.join(config['data']['model_save_dir'], config['data']['model_name'])
    best_model_pattern = os.path.join(model_path, 'best_model_epoch_*.pth')
    last_model_pattern = os.path.join(model_path, 'last_checkpoint_epoch*.pth')
   
    # Check and load the best or last model checkpoint
    best_model_files = glob.glob(best_model_pattern)
    last_model_files = glob.glob(last_model_pattern)  
    

    if best_model_files:
        # Load the best model based on the most recent epoch
        best_model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_model_path = best_model_files[-1]
        best_epoch = int(best_model_path.split('_')[-1].split('.')[0])
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    elif last_model_files:
        # Load the last checkpoint model
        last_model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        last_model_path = last_model_files[-1]
        best_epoch = int(last_model_path.split('_')[-1].split('.')[0])
        print(f"Loading last checkpoint model from {last_model_path}")
        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError("No best model or last model file found.")

    # Define directories for saving test examples and evaluation metrics
    test_examples_dir = os.path.join('test_eval', config['data']['model_name'], 'test_examples')
    test_eval_dir = os.path.join('test_eval', config['data']['model_name'], 'test_metrics', f"epoch_{best_epoch}")
    
    os.makedirs(test_examples_dir, exist_ok=True)
    os.makedirs(test_eval_dir, exist_ok=True)

    # Save top 10 best and worst examples from the test set
    save_top_and_bottom_examples(model, test_loader, test_examples_dir, device, best_epoch=best_epoch)

    # Compute the confusion matrix for the test data
    test_cm = compute_confusion_matrix(model, test_loader, device)

    # Plot and save the confusion matrix
    plot_confusion_matrix(test_cm, classes=['HP', 'SSA'], title='Confusion Matrix - Test Data', save_path=os.path.join(test_eval_dir, 'test_confusion_matrix.png'))
    plt.close()  # Ensure the plot is closed after saving

    # Initialize lists to store true labels, predictions, and probabilities
    all_labels = []
    all_preds = []
    all_probs = []

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Handle the case where outputs may be a tensor (ResNet) or an object with logits (ViT)
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                raise ValueError("Model output format not recognized")
            probs = nn.functional.softmax(logits, dim=1)[:, 1]  # Probability of class 1
            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu())
            all_preds.extend(preds.cpu().numpy())

    # Calculate and print evaluation metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_auc = roc_auc_score(all_labels, all_probs)
    test_precision = precision_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    # Save evaluation metrics to a text file
    with open(os.path.join(test_eval_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=['HP', 'SSA']))

if __name__ == '__main__':
    main()
