from datasets.tinyface.dataloader import get_eval_loaders, get_test_loaders, DatasetType
from tqdm import tqdm

def evaluate_model(model, dataset_type: DatasetType):
    if dataset_type == DatasetType.EVAL:
        gallery_loader, probe_loader = get_eval_loaders()
    elif dataset_type == DatasetType.TEST:
        gallery_loader, probe_loader = get_test_loaders()
    else:
        raise ValueError("Invalid dataset type")
    
    # Perform model evaluation using the gallery and probe loaders
    gallery_features = []
    probe_features = []

    for img, label in tqdm(gallery_loader, desc="Gallery features"):
        features = model(img)
        gallery_features.append(features)

    for img, label in tqdm(probe_loader, desc="Probe features"):
        features = model(img)
        probe_features.append(features)

    

