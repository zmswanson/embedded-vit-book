from datasets.tinyface.dataloader import get_eval_loaders, get_test_loaders
from datasets.tinyface.dataloader import DatasetType
from .probe_gallery import MultiLabelMethod
from .probe_gallery import get_batch_features, get_cosine_similarity
from .probe_gallery import get_rank_k_accuracy, get_roc_metrics

from tqdm import tqdm

def evaluate_model_rank_k(
    tinyface_data_path, model, dataset_type: DatasetType, rank_k: int = 5,
    multi_label_method: MultiLabelMethod = MultiLabelMethod.ALL
):
    """
    Helper fuction to quickly evaluate rank-k accuracy on TinyFace eval/test datasets.
    1. Loads the appropriate TinyFace dataset (eval or test) using dataloaders.
    2. Extracts features for gallery and probe images using the provided model.
    3. Computes cosine similarity between probe and gallery features.
    4. Calculates rank-k accuracy based on the similarity scores and labels.

    :param tinyface_data_path: Path to the TinyFace dataset.
    :type tinyface_data_path: str
    :param model: The model used for feature extraction.
    :type model: torch.nn.Module
    :param dataset_type: Type of dataset to evaluate (eval or test).
    :type dataset_type: DatasetType
    :param rank_k: The 'k' in rank-k accuracy. Defaults to 5.
    :type rank_k: int, optional
    :param multi_label_method: Method to handle multi-label gallery entries.
        Defaults to MultiLabelMethod.ALL.
    :type multi_label_method: MultiLabelMethod, optional

    :return: Rank-k accuracy values, where k ranges from 1 to rank_k.
    :rtype: list[float]
    """
    if dataset_type == DatasetType.EVAL:
        gallery_loader, probe_loader = get_eval_loaders(tinyface_data_path)
    elif dataset_type == DatasetType.TEST:
        gallery_loader, probe_loader = get_test_loaders(tinyface_data_path)
    else:
        raise ValueError("Invalid dataset type")
    
    # Perform model evaluation using the gallery and probe loaders
    gallery_features, gallery_labels = get_batch_features(model, gallery_loader)
    probe_features, probe_labels = get_batch_features(model, probe_loader)
    cosine_sim_matrix = get_cosine_similarity(probe_features, gallery_features)
    rank_k = get_rank_k_accuracy(
        cosine_sim_matrix, probe_labels, gallery_labels, k=rank_k,
        multi_label_method=multi_label_method
    )

    return rank_k


def evaluate_model_roc(
    tinyface_data_path, model, dataset_type: DatasetType, return_dict: bool = False,
    multi_label_method: MultiLabelMethod = MultiLabelMethod.ALL
):
    """
    Helper fuction to quickly evaluate ROC metrics on TinyFace eval/test datasets.
    1. Loads the appropriate TinyFace dataset (eval or test) using dataloaders.
    2. Extracts features for gallery and probe images using the provided model.
    3. Computes cosine similarity between probe and gallery features.
    4. Calculates ROC metrics including TPR, FPR, AUC, mAP, EER, and TPR@FPR=(1%, 5%).

    :param tinyface_data_path: Path to the TinyFace dataset.
    :type tinyface_data_path: str
    :param model: The model used for feature extraction.
    :type model: torch.nn.Module
    :param dataset_type: Type of dataset to evaluate (eval or test).
    :type dataset_type: DatasetType
    :param return_dict: Whether to return metrics as a dictionary. Defaults to False.
    :type return_dict: bool, optional
    :param multi_label_method: Method to handle multi-label gallery entries.
        Defaults to MultiLabelMethod.ALL.
    :type multi_label_method: MultiLabelMethod, optional

    :return: If return_dict is True, returns a dictionary with ROC metrics.
        Otherwise, returns a tuple of (fpr, tpr, thresholds, auc, mAP, eer,
        eer_threshold, tpr_at_fpr_1pct, tpr_at_fpr_5pct).
    :rtype: dict or tuple
    """
    if dataset_type == DatasetType.EVAL:
        gallery_loader, probe_loader = get_eval_loaders(tinyface_data_path)
    elif dataset_type == DatasetType.TEST:
        gallery_loader, probe_loader = get_test_loaders(tinyface_data_path)
    else:
        raise ValueError("Invalid dataset type")
    
    # Perform model evaluation using the gallery and probe loaders
    gallery_features, gallery_labels = get_batch_features(model, gallery_loader)
    probe_features, probe_labels = get_batch_features(model, probe_loader)
    cosine_sim_matrix = get_cosine_similarity(probe_features, gallery_features)
    
    roc_metrics = get_roc_metrics(
        cosine_sim_matrix, probe_labels, gallery_labels, return_dict=return_dict,
        multi_label_method=multi_label_method
    )

    return roc_metrics

if __name__ == "__main__":
    from torchvision import models
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pre-trained model without the final classification layer
    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last layer
    model.eval()
    
    model.to(device)

    test_path = "/mnt/data/biometrics/tinyface/"

    print("\nEvaluating on TinyFace EVAL dataset:")
    rank_k_eval = evaluate_model_rank_k(
        test_path, model, DatasetType.EVAL, rank_k=5,
        multi_label_method=MultiLabelMethod.ALL
    )
    roc_eval = evaluate_model_roc(
        test_path, model, DatasetType.EVAL, return_dict=True,
        multi_label_method=MultiLabelMethod.ALL
    )
    for key, value in roc_eval.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

    print(f"TinyFace EVAL Rank-1 to Rank-5 accuracies: {[f'{acc:.2f}' for acc in rank_k_eval]}\n")

    print("Evaluating on TinyFace TEST dataset:")
    rank_k_test = evaluate_model_rank_k(
        test_path, model, DatasetType.TEST, rank_k=5, 
        multi_label_method=MultiLabelMethod.ALL
    )
    roc_test = evaluate_model_roc(
        test_path, model, DatasetType.TEST, return_dict=True, 
        multi_label_method=MultiLabelMethod.ALL
    )
    for key, value in roc_test.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

    print(f"TinyFace TEST Rank-1 to Rank-5 accuracies: {[f'{acc:.2f}' for acc in rank_k_test]}\n")
