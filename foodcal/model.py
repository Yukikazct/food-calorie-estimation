"""Local MobileNetV3 initialization and versioned food-classifier loading."""

from pathlib import Path
import re

from PIL import ImageOps
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


ARCHITECTURE = "mobilenet_v3_small"
NORMALIZATION = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def resolve_device(device="auto"):
    if isinstance(device, torch.device):
        device = str(device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps or cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    return torch.device(device)


def validate_classes(class_names):
    if (not isinstance(class_names, (list, tuple)) or len(class_names) < 2
            or any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                   for name in class_names)
            or len(set(class_names)) != len(class_names)):
        raise ValueError("class_names must be an ordered list of unique class identifiers")
    return tuple(class_names)


def _prepare_image(image):
    return ImageOps.exif_transpose(image).convert("RGB")


def image_transform(input_size=224, augment=False):
    """Identical center crop in both controls; augmentation changes color only."""
    if type(input_size) is not int or input_size < 32:
        raise ValueError("input_size must be an integer of at least 32")
    steps = [transforms.Lambda(_prepare_image),
             transforms.Resize(round(input_size * 256 / 224), interpolation=transforms.InterpolationMode.BILINEAR),
             transforms.CenterCrop(input_size)]
    if augment:
        steps.append(transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.02))
    steps.extend([transforms.ToTensor(), transforms.Normalize(**NORMALIZATION)])
    return transforms.Compose(steps)


def build_model(num_classes, pretrained_weights=None):
    """Load only a supplied local ImageNet state, then initialize a new food head."""
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2")
    model = mobilenet_v3_small(weights=None)
    if pretrained_weights is not None:
        path = Path(pretrained_weights)
        if not path.is_file():
            raise FileNotFoundError(f"Local pretrained weights are missing: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def load_classifier(checkpoint, device="cpu"):
    """ImageNet weights cannot be passed off as a trained food classifier."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format") != "foodcal-classifier"
            or payload.get("version") != 1 or payload.get("trained") is not True
            or payload.get("architecture") != ARCHITECTURE):
        raise ValueError("Expected a trained foodcal classifier checkpoint, not ImageNet initialization")
    classes = validate_classes(payload.get("class_names"))
    translated = payload.get("class_names_zh")
    if not isinstance(translated, list) or len(translated) != len(classes) or any(not isinstance(s, str) or not s for s in translated):
        raise ValueError("Checkpoint class translations disagree with its ordered classes")
    if payload.get("normalization") != NORMALIZATION:
        raise ValueError("Unsupported checkpoint preprocessing")
    image_transform(payload.get("input_size"))
    model = build_model(len(classes))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(resolve_device(device)).eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata
