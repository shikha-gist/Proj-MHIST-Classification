from .resnet_model import get_resnet_model
from .vit_model import get_vit_model

def get_model(architecture, pretrained, num_classes, fine_tune_layers=-1, **kwargs):
    """
    Returns a model based on the specified architecture. Supports ResNet and ViT models.
    
    Args:
    - architecture (str): The model architecture to use ('resnet18', 'resnet34', 'vit', etc.).
    - pretrained (bool): Whether to use a pretrained model.
    - num_classes (int): Number of output classes.
    - fine_tune_layers (int): Number of layers to fine-tune. 
                              -1 means all layers are fine-tuned, 
                              0 means only the classification layer is fine-tuned,
                              any positive integer fine-tunes that many layers from the end.
    - **kwargs: Additional keyword arguments for specific model configurations.
    
    Returns:
    - model (torch.nn.Module): The initialized model.
    
    Raises:
    - ValueError: If an unsupported architecture is specified.
    """
    
    if architecture.startswith('resnet'):
        # If the architecture is a ResNet variant, return a ResNet model
        return get_resnet_model(architecture, pretrained, num_classes, fine_tune_layers)
    
    elif architecture.startswith('vit'):
        # If the architecture is a ViT variant, extract relevant ViT-specific parameters
        vit_variant = kwargs.get("variant", "base")
        vit_params = {
            "hidden_size": kwargs.get("hidden_size", 768),
            "num_attention_heads": kwargs.get("num_attention_heads", 12),
            "intermediate_size": kwargs.get("intermediate_size", 3072),
            "num_hidden_layers": kwargs.get("num_hidden_layers", 12)
        }
        return get_vit_model(variant=vit_variant, pretrained=pretrained, num_classes=num_classes, fine_tune_layers=fine_tune_layers, **vit_params)
    
    else:
        # Raise an error for unsupported architectures
        raise ValueError(f"Unsupported architecture: {architecture}")
