import torch
import torch.nn.functional as F
from tqdm import tqdm
from enum import Enum
from sklearn.metrics import roc_curve, average_precision_score, roc_auc_score
import numpy as np

class MultiLabelMethod(Enum):
    ALL = "all" # Use all the gallery labels for matching
    MEAN = "mean" # Use the mean similarity score for each unique gallery label
    MAX = "max" # Use the max similarity score for each unique gallery label

def get_batch_features(model, data_loader):
    # Get the device from the model's parameters
    device = next(model.parameters()).device.type
    print(f"{__name__} Using device: {device}")

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, labels_batch in tqdm(data_loader, desc="Extracting features"):
            images, labels_batch = images.to(device), labels_batch.to(device)

            output = model(images)
            output = output.flatten(start_dim=1)
            features.append(output.detach().cpu())
            labels.append(labels_batch.detach().cpu())
            
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def get_cosine_similarity(
    probe_features, gallery_features, probe_batch_size=256, device='cuda'
):
    """
    Compute cosine similarity between probe and gallery features. Returns a tensor
    of shape (num_probe_samples, num_gallery_samples) where each element is the
    cosine similarity between the i-th probe feature and the j-th gallery feature.
    
    :param probe_features: Features for the probe samples of shape
        (num_probe_samples, feature_dim)
    :type probe_features: torch.Tensor or np.ndarray
    :param gallery_features: Features for the gallery samples of shape
        (num_gallery_samples, feature_dim)
    :type gallery_features: torch.Tensor or np.ndarray
    :param probe_batch_size: Batch size for processing probe features to manage
        memory usage (default: 256). Use -1 to process all at once.
    :type probe_batch_size: int
    :param device: Device to perform computations on (default: 'cuda').
    :type device: str

    :return: Cosine similarity matrix of shape (num_probe_samples, num_gallery_samples)
    :rtype: torch.Tensor
    """
    if not isinstance(probe_features, torch.Tensor):
        probe_features = torch.tensor(probe_features)
    if not isinstance(gallery_features, torch.Tensor):
        gallery_features = torch.tensor(gallery_features)

    cos_sims = []

    if probe_batch_size == -1:
        probe_batch_size = probe_features.shape[0]

    gallery_features = gallery_features.to(device)
    probe_features = probe_features.to(device)

    if gallery_features.ndim == 1:
        gallery_features = gallery_features.unsqueeze(0)

    if probe_features.ndim == 1:
        probe_features = probe_features.unsqueeze(0)
    
    for probe_batch in tqdm(
        torch.split(probe_features, probe_batch_size),
        desc="Computing cosine similarities"
    ):
        # Efficiently compute cosine similarity using matrix operations with
        # broadcasting of norms and element-wise division of dot product results
        cos_sim_batch = (probe_batch @ gallery_features.T) / (
            torch.linalg.norm(probe_batch,ord=2, dim=1, keepdim=True) *
            torch.linalg.norm(gallery_features, ord=2, dim=1, keepdim=True).T
        )

        cos_sims.append(cos_sim_batch.detach().cpu())

    return torch.cat(cos_sims, dim=0)


def reduce_multi_label_similarities(
        cosine_similarities, gallery_labels, multi_label_method: MultiLabelMethod
    ):
    """
    Reduces the cosine similarity matrix based on the specified multi-label method.
    For each unique gallery label, it aggregates the similarity scores using the
    specified method (ALL, MEAN, MAX).

    :param cosine_similarities: Matrix of cosine similarity scores with shape
        (num_probe_samples, num_gallery_samples)
    :type cosine_similarities: torch.Tensor
    :param gallery_labels: Labels for the gallery samples
    :type gallery_labels: torch.Tensor
    :param multi_label_method: Method to reduce multi-label similarities
    :type multi_label_method: MultiLabelMethod

    :return: Reduced cosine similarity matrix of shape
      (num_probe_samples, num_unique_gallery_labels) and the unique gallery labels
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    """
    unique_gallery_labels = torch.unique(gallery_labels)
    num_probe_samples = cosine_similarities.shape[0]
    reduced_similarities = torch.zeros((num_probe_samples, len(unique_gallery_labels)))

    for i, gallery_label in enumerate(unique_gallery_labels):
        mask = (gallery_labels == gallery_label)
        sims_for_label = cosine_similarities[:, mask]
        if multi_label_method == MultiLabelMethod.ALL:
            return cosine_similarities, gallery_labels
        elif multi_label_method == MultiLabelMethod.MEAN:
            reduced_similarities[:, i] = sims_for_label.mean(dim=1)
        elif multi_label_method == MultiLabelMethod.MAX:
            reduced_similarities[:, i] = sims_for_label.max(dim=1).values
        else:
            raise ValueError(f"Invalid multi-label method: {multi_label_method}")

    return reduced_similarities, unique_gallery_labels


def get_rank_k_accuracy(
        cosine_similarities, probe_labels, gallery_labels, k=10,
        multi_label_method=MultiLabelMethod.ALL
    ):
    """
    Takes in a cosine similarity matrix of shape (num_probe_samples, num_gallery_samples)
    and the lables for the probe and gallery samples. For each probe sample, it
    finds the top-k indices in the gallery based on cosine similarity, checks if
    the labels match, and returns the rank-k accuracies up to k. For example, if
    k=5, it returns a list of 5 values where the i-th value is the percentage of
    probe samples that had a correct match in the top-(i+1) gallery samples.
    
    :param cosine_similarities: Matrix of cosine similarity scores with shape
        (num_probe_samples, num_gallery_samples)
    :type cosine_similarities: torch.Tensor
    :param probe_labels: Labels for the probe samples
    :type probe_labels: torch.Tensor
    :param gallery_labels: Labels for the gallery samples
    :type gallery_labels: torch.Tensor
    :param k: Maximum rank to compute accuracy for
    :type k: int

    :return: List of rank-k accuracies up to and including k
    :rtype: List[float]
    """
    if multi_label_method != MultiLabelMethod.ALL:
        cosine_similarities, gallery_labels = reduce_multi_label_similarities(
            cosine_similarities, gallery_labels, multi_label_method
        )

    num_probe_samples = cosine_similarities.shape[0]
    topk_indices = torch.topk(cosine_similarities, k=k, dim=1).indices

    rank_k_accuracies = []
    for rank in range(1, k + 1):
        correct_matches = 0
        for i in range(num_probe_samples):
            topk_gallery_indices = topk_indices[i, :rank]
            if probe_labels[i] in gallery_labels[topk_gallery_indices]:
                correct_matches += 1
        accuracy = correct_matches / num_probe_samples
        rank_k_accuracies.append(accuracy)

    return rank_k_accuracies


def get_roc_metrics(
    cosine_similarities, probe_labels, gallery_labels, return_dict=False,
    multi_label_method=MultiLabelMethod.ALL
):
    """
    Computes ROC metrics including FPR, TPR, thresholds, AUC, mAP, EER, and TPR at
    specific FPR levels from the cosine similarity matrix and corresponding labels.

    :param cosine_similarities: Matrix of cosine similarity scores with shape
        (num_probe_samples, num_gallery_samples)
    :type cosine_similarities: torch.Tensor
    :param probe_labels: Labels for the probe samples
    :type probe_labels: torch.Tensor
    :param gallery_labels: Labels for the gallery samples
    :type gallery_labels: torch.Tensor
    :param return_dict: Whether to return results as a dictionary (default: False)
    :type return_dict: bool
    :param multi_label_method: Method to reduce multi-label similarities
    :type multi_label_method: MultiLabelMethod

    :return: If return_dict is True, returns a dictionary with ROC metrics.
        Otherwise, returns a tuple of (fpr, tpr, thresholds, auc, mAP, eer,
        eer_threshold, tpr_at_fpr_1pct, tpr_at_fpr_5pct).
    :rtype: dict or tuple
    """
    if multi_label_method != MultiLabelMethod.ALL:
        cosine_similarities, gallery_labels = reduce_multi_label_similarities(
            cosine_similarities, gallery_labels, multi_label_method
        )
    
    id_matches = (
        probe_labels.unsqueeze(1) == gallery_labels.unsqueeze(0)
    ).flatten().numpy()
    sim_scores = cosine_similarities.flatten().numpy()

    fpr, tpr, thresholds = roc_curve(id_matches, sim_scores)
    auc = roc_auc_score(id_matches, sim_scores)

    # calculate mean average precision (mAP) across all probe samples
    avg_precs = []
    for i in range(probe_labels.shape[0]):
        ap = average_precision_score(
            (gallery_labels == probe_labels[i]).numpy(),
            cosine_similarities[i].numpy()
        )
        avg_precs.append(ap)
    mAP = np.mean(avg_precs)

    # calculate eer from fpr and tpr
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    # calculate tpr at fpr = 0.01 and fpr = 0.05
    tpr_at_fpr_1pct= tpr[np.nanargmin(np.absolute((fpr - 0.01)))]
    tpr_at_fpr_5pct = tpr[np.nanargmin(np.absolute((fpr - 0.05)))]

    if return_dict:
        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc,
            "mAP": mAP,
            "eer": eer,
            "eer_threshold": eer_threshold,
            "tpr_at_fpr_1pct": tpr_at_fpr_1pct,
            "tpr_at_fpr_5pct": tpr_at_fpr_5pct,
        }

    return fpr, tpr, thresholds, auc, mAP, eer, eer_threshold, tpr_at_fpr_1pct, tpr_at_fpr_5pct


if __name__ == "__main__":
    from torchvision import models

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pre-trained model without the final classification layer
    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last layer
    model.eval()
    
    model.to(device)

    print("\n******* Starting feature extraction functional test ********\n")

    # Generate dummy data for testing
    dummy_images1 = torch.randn(128, 3, 224, 224)
    dummy_images2 = torch.randn(256, 3, 224, 224)
    dummy_labels1 = torch.randint(0, 10, (128,))
    dummy_labels2 = torch.randint(0, 10, (256,))

    dataset1 = torch.utils.data.TensorDataset(dummy_images1, dummy_labels1)
    dataset2 = torch.utils.data.TensorDataset(dummy_images2, dummy_labels2)

    dummy_loader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=32, shuffle=False, num_workers=4
    )

    dummy_loader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=32, shuffle=False, num_workers=4
    )

    features1, labels1 = get_batch_features(model, dummy_loader1)
    features2, labels2 = get_batch_features(model, dummy_loader2)

    print(f"Features shape: {features1.shape}, {features2.shape}")
    print(f"Labels shape: {labels1.shape}, {labels2.shape}")

    cosine_similarities = get_cosine_similarity(features1, features2, device=device)
    print(f"Cosine similarities shape: {cosine_similarities.shape}")

    print("\n******* Starting cosine similarity functional test ********\n")

    dummy_embedding = torch.randn(1, 512)
    orthogonal_embedding = torch.randn(1, 512)
    orthogonal_embedding -= (
        ((dummy_embedding @ orthogonal_embedding.T) / 
        (dummy_embedding @ dummy_embedding.T)) * dummy_embedding
    )
    identical_embedding = dummy_embedding.clone()
    opposite_embedding = -dummy_embedding.clone()
    random_embedding = torch.randn(1, 512)

    cos_same = get_cosine_similarity(dummy_embedding, identical_embedding, device=device)
    cos_opp = get_cosine_similarity(dummy_embedding, opposite_embedding, device=device)
    cos_ortho = get_cosine_similarity(dummy_embedding, orthogonal_embedding, device=device)
    cos_rand = get_cosine_similarity(dummy_embedding, random_embedding, device=device)

    print(f"Cosine similarity (same): {cos_same.item():.4f} (expected ~1.0)")
    print(f"Cosine similarity (opposite): {cos_opp.item():.4f} (expected ~-1.0)")
    print(f"Cosine similarity (orthogonal): {cos_ortho.item():.4f} (expected ~0.0)")
    print(f"Cosine similarity (random): {cos_rand.item():.4f}")

    print("\n******* Starting top-k matches functional test ********\n")

    probe_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    gallery_labels = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    ])

        # gallery 0 is ranked 1 for probe 0 [1, 1, 1, 1, 1]
        # gallery 1 is ranked 2 for probe 1 [0, 1, 1, 1, 1]
        # gallery 2 is ranked 3 for probe 2 [0, 0, 1, 1, 1]
        # gallery 3 is ranked 4 for probe 3 [0, 0, 0, 1, 1]
        # gallery 4 is ranked 5 for probe 4 [0, 0, 0, 0, 1]
        # gallery 5 is ranked 1 for probe 5 [1, 1, 1, 1, 1]
        # gallery 6 is ranked 1 for probe 6 [1, 1, 1, 1, 1]
        # gallery 7 is ranked 2 for probe 7 [0, 1, 1, 1, 1]
        # gallery 8 is ranked 3 for probe 8 [0, 0, 1, 1, 1]
        # gallery 9 is ranked 7 for probe 9 [0, 0, 0, 0, 0]
        #                              sum: [3, 5, 7, 8, 9]
        # Expected accuracies: [0.3, 0.5, 0.7, 0.8, 0.9]
    cosine_sim_matrix = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
        [1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
        [1.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
        [1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.3, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,],
    ])

    reduced_sim_matrix, reduced_gallery_labels = reduce_multi_label_similarities(
        cosine_sim_matrix, gallery_labels, multi_label_method=MultiLabelMethod.ALL
    )

    print("[*] Test using all gallery labels:")
    print("Original shape:", cosine_sim_matrix.shape, "| Reduced shape:", reduced_sim_matrix.shape)

    rank_k_accuracies = get_rank_k_accuracy(reduced_sim_matrix, probe_labels, gallery_labels, k=5)
    print(f"Rank-<1:5> accuracies: {[f'{acc:.2f}' for acc in rank_k_accuracies]}\n")

    print("\n[*] Test using mean similarity for each unique gallery label:")
    reduced_sim_matrix, reduced_gallery_labels = reduce_multi_label_similarities(
        cosine_sim_matrix, gallery_labels, multi_label_method=MultiLabelMethod.MEAN
    )
    print("Original shape:", cosine_sim_matrix.shape, "| Reduced shape:", reduced_sim_matrix.shape)
    print(reduced_sim_matrix.numpy().round(2),"\n")

    rank_k_accuracies = get_rank_k_accuracy(reduced_sim_matrix, probe_labels, reduced_gallery_labels, k=5)
    print(f"Rank-<1:5> accuracies: {[f'{acc:.2f}' for acc in rank_k_accuracies]}\n")

    print("\n[*] Test using max similarity for each unique gallery label:")
    reduced_sim_matrix, reduced_gallery_labels = reduce_multi_label_similarities(
        cosine_sim_matrix, gallery_labels, multi_label_method=MultiLabelMethod.MAX
    )
    print("Original shape:", cosine_sim_matrix.shape, "| Reduced shape:", reduced_sim_matrix.shape)
    print(reduced_sim_matrix.numpy().round(2),"\n")

    rank_k_accuracies = get_rank_k_accuracy(reduced_sim_matrix, probe_labels, reduced_gallery_labels, k=5)
    print(f"Rank-<1:5> accuracies: {[f'{acc:.2f}' for acc in rank_k_accuracies]}\n")

    roc_metrics = get_roc_metrics(reduced_sim_matrix, probe_labels, reduced_gallery_labels, return_dict=True)
    for key, value in roc_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    
    print("\n******* All functional tests completed ********\n")