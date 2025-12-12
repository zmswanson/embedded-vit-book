from datasets.tinyface.dataloader import get_eval_loaders, get_test_loaders, DatasetType
from .probe_gallery import MultiLabelMethod, get_batch_features, get_cosine_similarity, get_rank_k_accuracy

from tqdm import tqdm

def evaluate_model(
    tinyface_data_path, model, dataset_type: DatasetType, rank_k: int = 5,
    multi_label_method: MultiLabelMethod = MultiLabelMethod.ALL
):
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
    rank_k_eval = evaluate_model(
        test_path, model, DatasetType.EVAL, rank_k=5, multi_label_method=MultiLabelMethod.ALL
    )
    print(f"TinyFace EVAL Rank-1 to Rank-5 accuracies: {[f'{acc:.2f}' for acc in rank_k_eval]}\n")

    print("Evaluating on TinyFace TEST dataset:")
    rank_k_test = evaluate_model(
        test_path, model, DatasetType.TEST, rank_k=5, multi_label_method=MultiLabelMethod.ALL
    )
    print(f"TinyFace TEST Rank-1 to Rank-5 accuracies: {[f'{acc:.2f}' for acc in rank_k_test]}\n")
