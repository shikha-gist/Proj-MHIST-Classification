import torch
from transformers import ViTForImageClassification, ViTConfig

"""
This script defines a function to create and configure a Vision Transformer (ViT) model with options for 
pretrained weights, custom architecture configurations, and fine-tuning specific layers. It supports standard 
ViT variants (base, large, huge) as well as custom configurations for research and development purposes.
"""

def adjust_hidden_size(num_attention_heads, desired_hidden_size=768):
    """
    Adjusts the hidden_size to be the closest multiple of num_attention_heads.

    Args:
    - num_attention_heads (int): The number of attention heads in the transformer.
    - desired_hidden_size (int): The target hidden size to adjust.

    Returns:
    - adjusted_hidden_size (int): The closest multiple of num_attention_heads to the desired hidden size.
    """
    # Find the closest multiple of num_attention_heads to the desired_hidden_size
    adjusted_hidden_size = max(num_attention_heads, (desired_hidden_size // num_attention_heads) * num_attention_heads)
    
    # Ensure it's a multiple
    if adjusted_hidden_size % num_attention_heads != 0:
        adjusted_hidden_size = (desired_hidden_size // num_attention_heads + 1) * num_attention_heads
    
    return adjusted_hidden_size

def get_vit_model(variant='base', pretrained=True, num_classes=2, fine_tune_layers=-1, hidden_size=768, num_attention_heads=12, intermediate_size=3072, num_hidden_layers=12):
    """
    Get a Vision Transformer (ViT) model with options for fine-tuning layers and custom hyperparameters.
    
    Args:
    - variant (str): The ViT variant to use ('base', 'large', 'huge', 'custom-vit').
    - pretrained (bool): Whether to load pretrained weights.
    - num_classes (int): The number of output classes.
    - fine_tune_layers (int): The number of layers to fine-tune. 
                              0 means only the final layer is trained, 
                              -1 means all layers are fine-tuned, 
                              and any positive integer fine-tunes that many layers from the end.
    - hidden_size (int): The size of the hidden layers.
    - num_attention_heads (int): The number of attention heads in the transformer.
    - intermediate_size (int): The size of the "intermediate" (i.e., feed-forward) layer in the transformer.
    - num_hidden_layers (int): The number of hidden layers in the transformer.
    
    Returns:
    - model (torch.nn.Module): The Vision Transformer (ViT) model.
    """
    
    # Define model configurations for different variants
    # Map for standard model variants
    model_name_map = {
        'base': 'google/vit-base-patch16-224',
        'large': 'google/vit-large-patch16-224',
        'huge': 'google/vit-huge-patch14-224'
    }

    # Check if variant is a standard pre-trained model
    if variant in model_name_map:
        model_name = model_name_map.get(variant.lower())
        if pretrained:
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True  # Handle mismatched classifier sizes
            )
        else:
            # Use default configuration for the selected variant without pre-trained weights
            config = ViTConfig.from_pretrained(
                model_name,
                num_labels=num_classes
            )
            model = ViTForImageClassification(config)
    
    # Handle custom ViT configurations
    elif variant == 'custom-vit' and not pretrained:
        # Adjust the hidden size based on num_attention_heads
        hidden_size = adjust_hidden_size(num_attention_heads, desired_hidden_size=hidden_size)
        config = ViTConfig(
            hidden_size=hidden_size,              # Adjusted hidden size
            num_attention_heads=num_attention_heads,  # Number of attention heads
            intermediate_size=intermediate_size,  # Typically this would be 4x the hidden size
            num_hidden_layers=num_hidden_layers,
            num_labels=num_classes
        )
        model = ViTForImageClassification(config)
    
    else:
        raise ValueError(f"Unsupported variant: {variant}. Supported variants: {list(model_name_map.keys()) + ['custom-vit']}")
    
    # Fine-tuning specific layers
    if pretrained:
        if fine_tune_layers == -1:
            # Fine-tune all layers
            for param in model.parameters():
                param.requires_grad = True
        elif fine_tune_layers > 0:
            # Fine-tune the last 'fine_tune_layers' layers
            total_layers = model.config.num_hidden_layers
            layers_to_fine_tune = fine_tune_layers

            for i, layer in enumerate(model.vit.encoder.layer):
                if i >= total_layers - layers_to_fine_tune:
                    for param in layer.parameters():
                        param.requires_grad = True
                else:
                    for param in layer.parameters():
                        param.requires_grad = False

            # Always fine-tune the classification head
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
            # Fine-tune only the final classification layer
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
                
    return model
