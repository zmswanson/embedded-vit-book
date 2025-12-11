from operator import mod
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
            features.append(output.squeeze().cpu())
            labels.append(labels_batch.cpu())
            
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def get_cosine_similarity(probe_features, gallery_features, device='cuda'):
    """
    Compute cosine similarity between probe and gallery features. Returns a tensor
    of shape (num_probe_samples, num_gallery_samples) where each element is the cosine
    similarity between the i-th probe feature and the j-th gallery feature.
    
    :param probe_features: Description
    :param gallery_features: Description
    :param device: Description
    """
    if not isinstance(probe_features, torch.Tensor):
        probe_features = torch.tensor(probe_features)
    if not isinstance(gallery_features, torch.Tensor):
        gallery_features = torch.tensor(gallery_features)

    probe_features.to(device)
    gallery_features.to(device)

    return F.cosine_similarity(probe_features.unsqueeze(1), gallery_features.unsqueeze(0), dim=2)


def get_top_k_matches(cosine_similarities, probe_labels, gallery_labels, k=5):
    top_k_matches = []
    for i in range(len(probe_labels)):
        top_k_indices = torch.topk(cosine_similarities[i], k).indices.cpu().numpy()
        top_k_labels = [gallery_labels[j] for j in top_k_indices]
        top_k_matches.append(top_k_labels)

    return top_k_matches


if __name__ == "__main__":
    from torchvision import models

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pre-trained model without the final classification layer
    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last layer
    model.eval()
    
    model.to(device)

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

