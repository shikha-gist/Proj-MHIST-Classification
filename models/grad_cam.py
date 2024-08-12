import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {'SSA': 1, 'HP': 0}  # Mapping of labels to integers

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, self.dataframe.columns.get_loc('Image Name')])
        image = Image.open(img_name).convert("RGB")
        label_str = self.dataframe.iloc[idx, self.dataframe.columns.get_loc('Majority Vote Label')]

        # Convert label from string to integer using label_map
        label = self.label_map.get(label_str)
        if label is None:
            raise ValueError(f"Label '{label_str}' not found in label_map")

        if self.transform:
            image = self.transform(image)
        return image, label, img_name

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_layers()

    def _hook_layers(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()  # Extract the relevant tensor from the tuple
            else:
                self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_image)

        if hasattr(output, 'logits'):
            output = output.logits  # Extract the logits if using ViT or similar models

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[:, class_idx]
        target.backward()

        if self.gradients is None:
            raise RuntimeError("Gradients were not captured; ensure the backward hook is properly registered.")

        gradients = self.gradients.cpu().numpy()
        activations = self.activations.cpu().numpy()

        if len(gradients.shape) == 4:  # CNN case
            weights = np.mean(gradients, axis=(2, 3))  # Average across height and width
            heatmap = np.zeros(activations.shape[2:], dtype=np.float32)
            for i, w in enumerate(weights[0]):
                heatmap += w * activations[0, i, :, :]


        elif len(gradients.shape) == 3:  # Transformer case (ViT)
            weights = np.mean(gradients, axis=2)  # Average across the hidden size
            sequence_length = activations.shape[1]  # Sequence length (e.g., 197 for ViT)
            hidden_size = activations.shape[2]  # Hidden size (e.g., 1024)
            patch_activations = activations[:, 1:, :]  # Exclude the first token (CLS token)
            grid_size = int(np.sqrt(patch_activations.shape[1]))  # Recalculate grid_size, should be 14

            activations_reshaped = patch_activations.reshape((activations.shape[0], grid_size, grid_size, -1))
            # Generate the heatmap by applying weights to the reshaped activations
            heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
            for i in range(sequence_length - 1):  # Iterate over the patches, excluding the class token
                patch_weight = weights[0, i]
                heatmap += patch_weight * activations_reshaped[0, :, :, i % hidden_size]
        else:
            raise ValueError("Unexpected gradients shape: ", gradients.shape)

        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / heatmap.max() if heatmap.max() != 0 else heatmap
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, beta=0.1, colormap='jet'):
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size, Image.Resampling.LANCZOS)
        heatmap_resized = np.array(heatmap_resized) / 255.0
        heatmap_colored = plt.get_cmap(colormap)(heatmap_resized)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)
        overlayed_image = np.uint8(alpha * heatmap_colored + (1-alpha) * np.array(image))
        return Image.fromarray(overlayed_image), heatmap_resized
