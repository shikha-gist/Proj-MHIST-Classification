import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from utils.utils import denormalize
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageOps
from collections import Counter


def train_model(model, train_loader, val_loader, model_config, model_save_dir, device, start_epoch=0, resume=False):
    # Create model_save_dir if it doesn't exist
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    # Define loss function and optimizer
    if model_config['use_class_weights']:
        class_counts = Counter([train_loader.dataset[i][1] for i in range(len(train_loader.dataset))])
        class_weights = [1.0 / class_counts[cls] for cls in range(len(class_counts))]
        total = sum(class_weights)
        normalized_class_weights = [w / total for w in class_weights]
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(normalized_class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=model_config['step_size'], gamma=model_config['gamma'])

    # Initialize training history lists
    train_acc_history, val_acc_history = [], []
    train_auc_history, val_auc_history = [],[]
    train_loss_history, val_loss_history = [],[]
    learning_rates, training_logs = [], []


    # Resume training if specified
    if resume:
        checkpoints = [f for f in os.listdir(model_save_dir) if f.startswith('last_checkpoint_epoch')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by epoch number
            last_checkpoint = checkpoints[-1]  # Get the most recent checkpoint
            checkpoint_path = os.path.join(model_save_dir, last_checkpoint)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            if model_config['num_epochs'] == start_epoch:
                print(f"Model is already trained for {start_epoch} epochs.")
                print("Starting the evaluation on the best saved model:")
            else:            
                print(f"Resuming training from epoch {start_epoch + 1}")
            # Load existing logs
            logs_path = os.path.join(model_save_dir, 'training_logs.csv')
            if os.path.exists(logs_path):
                existing_logs = pd.read_csv(logs_path).to_dict('records')
                training_logs = existing_logs
                train_acc_history = [log['train_accuracy'] for log in training_logs]
                val_acc_history = [log['val_accuracy'] for log in training_logs]
                train_auc_history = [log['train_auc'] for log in training_logs]
                val_auc_history = [log['val_auc'] for log in training_logs]
                train_loss_history = [log['train_loss'] for log in training_logs]
                val_loss_history = [log['val_loss'] for log in training_logs]
                learning_rates = [log['learning_rate'] for log in training_logs]
        else:
            print(f"No checkpoint found in {model_save_dir}. Starting fresh training.")
    else:
        start_epoch = 0  # Start from scratch if not resuming
        print("Not asked for resuming, so starting fresh training.")

    # Training loop
    num_epochs = model_config['num_epochs']
    best_val_metric = 0.0  # Initialize the best validation metric (accuracy or AUC)
    best_model_path = None
    epochs_no_improve = 0  # Count epochs with no improvement

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        train_labels = []
        train_preds = []
        # train_outputs = []
        train_probs = []

        for images, labels, _ in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Handle the case for different model types (e.g., ResNet vs ViT)
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:  # 
                logits = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(preds.cpu().numpy())
            train_probs_n = nn.functional.softmax(logits, dim=1)[:, 1].detach()
            train_probs.extend(train_probs_n.cpu().numpy())

        train_accuracy = accuracy_score(train_labels, train_preds)
        train_auc = roc_auc_score(train_labels, train_probs)
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_labels = []
        val_preds = []
        val_probs = []
        val_running_loss = 0.0

        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # Handle the case for different model types (e.g., ResNet vs ViT)
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:  # 
                    logits = outputs.logits

                loss = criterion(logits, labels)
                val_running_loss += loss.item()
                _, preds = torch.max(logits, 1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_probs_n = nn.functional.softmax(logits, dim=1)[:, 1].detach()
                val_probs.extend(val_probs_n.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_loss = val_running_loss / len(val_loader)

        # Determine the metric to monitor
        if model_config['monitor_metric'].lower() == 'auc':
            current_metric = val_auc
        else:
            current_metric = val_accuracy

        # Check if there is an improvement
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            epochs_no_improve = 0  # Reset counter if there is an improvement
            best_model_path = os.path.join(model_save_dir, f'best_model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1  # Increment counter if no improvement
        # Append metrics to history
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)
        train_auc_history.append(train_auc)
        val_auc_history.append(val_auc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        new_log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'train_auc': train_auc,
            'val_accuracy': val_accuracy,
            'val_auc': val_auc,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_logs.append(new_log_entry)

        # Print training and validation metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}")

   
        # Save the current checkpoint
        last_model_path = os.path.join(model_save_dir, f'last_checkpoint_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_metric': best_val_metric,
            'training_logs': training_logs,
        }, last_model_path)

        # Remove previous checkpoint files
        for file in os.listdir(model_save_dir):
            if file.startswith("last_checkpoint") and file != os.path.basename(last_model_path):
                os.remove(os.path.join(model_save_dir, file))
            if file.startswith("best_model") and file != os.path.basename(best_model_path):
                os.remove(os.path.join(model_save_dir, file))



        # Early stopping condition
        if model_config['early_stopping'] and epochs_no_improve >= model_config['patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break       

        scheduler.step()  # Update the learning rate
    # Save training logs to a file (append mode)
    logs_path = os.path.join(model_save_dir, 'training_logs.csv')
    if os.path.exists(logs_path):
        existing_logs = pd.read_csv(logs_path).to_dict('records')
        if existing_logs[-1]['epoch'] < training_logs[0]['epoch']:
            training_logs = existing_logs + training_logs
    pd.DataFrame(training_logs).to_csv(logs_path, index=False)

    # Create directory for training curves if it doesn't exist
    curves_dir = os.path.join(model_save_dir, 'training_curves')
    if not os.path.exists(curves_dir):
        os.makedirs(curves_dir)

    # Plot accuracy, AUC, loss, and learning rate curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Train Accuracy')
    plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(curves_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_auc_history) + 1), train_auc_history, label='Train AUC')
    plt.plot(range(1, len(val_auc_history) + 1), val_auc_history, label='Validation AUC')
    plt.title('AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(os.path.join(curves_dir, 'auc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(curves_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, label='Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.savefig(os.path.join(curves_dir, 'learning_rate_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_top_and_bottom_examples(model, loader, save_dir, device, num_examples=10, best_epoch=None, s_type='Test Set:'):
    model.eval()
    all_images = []
    all_labels = []
    all_probs = []
    all_preds = []
    all_img_names = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{s_type} Evaluating for top, bottom, and medium examples"):
            if len(batch) == 4:  # Case where there are four items returned
                images, image_no_norms, labels, img_names = batch
            elif len(batch) == 3:  # Case where there are three items returned
                images, labels, img_names = batch
            else:
                raise ValueError("Unexpected number of items returned by the DataLoader")

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

            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())
            all_probs.extend(probs.cpu())
            all_preds.extend(preds.cpu())
            all_img_names.extend(img_names)

    all_images = torch.stack(all_images)
    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)
    all_preds = torch.tensor(all_preds)
    
    # Calculate overall performance metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    # Save performance metrics to a text file
    performance_metrics = {
        'Accuracy': accuracy,
        'AUC': auc
    }
    save_dir = os.path.join(save_dir, "epoch_" + str(best_epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'performance_metrics.txt'), 'w') as f:
        for key, value in performance_metrics.items():
            f.write(f'{key}: {value:.4f}\n')

    # Calculate differences for worst and medium examples
    differences = torch.abs(all_probs - all_labels.float())
    proximity_to_half = torch.abs(all_probs - 0.5)

    # Get indices for the worst examples (largest differences)
    _, worst_indices = torch.topk(differences, min(num_examples, len(differences)), largest=True)

    # Get indices for the best examples (smallest differences)
    _, best_indices = torch.topk(differences, min(num_examples, len(differences)), largest=False)

    # Get indices for medium examples (closest to 0.5)
    _, medium_indices = torch.topk(proximity_to_half, min(num_examples, len(proximity_to_half)), largest=False)

    def save_examples(indices, prefix):
        for i, idx in enumerate(indices):
            image = all_images[idx]
            label = all_labels[idx].item()
            prob = round(all_probs[idx].item(), 2)
            pred = all_preds[idx].item()
            img_name = os.path.basename(all_img_names[idx])

            image = denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = image.numpy().transpose((1, 2, 0))
            image = np.clip(image, 0, 1)

            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            img_pil = ImageOps.expand(img_pil, border=2, fill='green')

            class_names = {0: 'HP', 1: 'SSA'}
            text = f'True: {class_names[label]}, Pred: {class_names[pred]}, Prob: {prob:.2f}'
            if best_epoch is not None:
                text += f', Best Epoch: {best_epoch}'

            text_img = Image.new('RGB', (img_pil.width, 30), color=(173, 216, 230))
            draw = ImageDraw.Draw(text_img)
            font = ImageFont.load_default()
            draw.text((10, 10), text, font=font, fill=(0, 0, 0))

            combined_img = Image.new('RGB', (img_pil.width, img_pil.height + text_img.height))
            combined_img.paste(text_img, (0, 0))
            combined_img.paste(img_pil, (0, text_img.height))

            combined_img.save(os.path.join(save_dir, f'{prefix}_example_{i + 1}_{img_name}.png'))

    # Save the best, worst, and medium examples
    save_examples(best_indices, 'best')
    save_examples(worst_indices, 'worst')
    save_examples(medium_indices, 'medium')



def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close()

def compute_confusion_matrix(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Handle different output types for ResNet and ViT models
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits  # For ViT models, extract the logits
            
            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm


