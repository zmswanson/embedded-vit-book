import re
from typing import List, Optional, Tuple

def _infer_family(model_name: Optional[str]) -> str:
    """
    Infer a family primarily from the timm model name.
    Keep this simple/explicit to avoid surprises.
    """
    if not model_name:
        return "generic"  # safe default for transformer-like models

    mn = model_name.lower()

    if "mobilevit" in mn:
        return "mobilevit"
    if mn.startswith("pvt") or "pvt" in mn:
        return "pvt"
    if "swin" in mn:
        return "swin"
    if "levit" in mn:
        return "levit"
    if "efficientformer" in mn:
        return "efficientformer"
    if "convnext" in mn:
        return "convnext"
    if "resnet" in mn:
        return "resnet"
    if "efficientnet" in mn:
        return "efficientnet"
    if "mobilenet" in mn:
        return "mobilenet"
    if "deit" in mn or "vit" in mn or "beit" in mn or re.search(r"\bvit\b", mn):
        return "vit"

    return "generic"

