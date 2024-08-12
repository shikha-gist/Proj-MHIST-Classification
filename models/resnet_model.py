import torch
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

def get_resnet_model(architecture, pretrained, num_classes, fine_tune_layers=-1):
    # Map architecture names to torchvision models
    resnet_model_map = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152
    }
    
    # Check if the requested architecture is supported
    if architecture not in resnet_model_map:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported architectures: {list(resnet_model_map.keys())}")
    
    # Define weights mapping
    weights_map = {
        'resnet18': ResNet18_Weights.IMAGENET1K_V1,
        'resnet34': ResNet34_Weights.IMAGENET1K_V1,
        'resnet50': ResNet50_Weights.IMAGENET1K_V1,
        'resnet101': ResNet101_Weights.IMAGENET1K_V1,
        'resnet152': ResNet152_Weights.IMAGENET1K_V1
    }

    # Load the model with or without pretrained weights
    model = resnet_model_map[architecture](weights=weights_map[architecture] if pretrained else None)
    
    # Replace the final fully connected layer with a new one for the specified number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    if pretrained:
        if fine_tune_layers == -1:
            # Fine-tune all layers
            for param in model.parameters():
                param.requires_grad = True
        elif fine_tune_layers > 0:
            # Flatten model layers, including sub-layers in Sequential modules
            layers = []
            for name, module in model.named_children():
                if isinstance(module, torch.nn.Sequential):
                    for sub_name, sub_module in module.named_children():
                        layers.append(sub_module)
                else:
                    layers.append(module)
            
            # Fine-tune the last N layers
            for i, layer in enumerate(layers):
                if i >= len(layers) - fine_tune_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
                else:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            # Only fine-tune the final classification layer
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
    
    return model
