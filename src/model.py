import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

CLASSES = ['ship', 'background']

def build_model() -> nn.Module:
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.aux_classifier = None

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = DeepLabHead(2048, len(CLASSES))

    # model to device!

    return model

def load_model(path: str, device: str) -> nn.Module:
    model = build_model()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state']) # fixme: key is duplicated
    model.to(device)
    model.eval()

    return model
