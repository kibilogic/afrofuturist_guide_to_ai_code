import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np   665
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    Subset,
    random_split
)

# Configuration and Device Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-10 class names (in alphabetical order)
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Model Definitions 
class AdversarialDetector(nn.Module):
    def __init__(self, feature_extractor, detector_layers=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze extractor

        if detector_layers is None:
            detector_layers = [1280, 512, 128, 1]  # MobileNetV2 dim

        layers = []
        for i in range(len(detector_layers) - 1):
            layers.append(nn.Linear(
                detector_layers[i], detector_layers[i + 1]
            ))
            if i < len(detector_layers) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
        self.detector = nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
        detection_score = torch.sigmoid(self.detector(features))
        return detection_score


def train_detector(
    detector,
    clean_train_loader,
    adv_train_loader,
    clean_val_loader,
    adv_val_loader,
    epochs=10,
    weight_decay=1e-5
):
    optimizer = torch.optim.Adam(
        detector.parameters(),
        lr=0.001,
        weight_decay=weight_decay
    )
    criterion = nn.BCELoss()
    best_val_auc = -1.0
    best_threshold = 0.5

    for epoch in range(epochs):
        detector.train()
        total_loss = 0.0

        for (clean_batch, _), (adv_batch, _) in zip(
            clean_train_loader, adv_train_loader
        ):
            clean_batch = clean_batch.to(device)
            adv_batch = adv_batch.to(device)

            optimizer.zero_grad()

            clean_scores = detector(clean_batch)
            clean_labels = torch.zeros_like(clean_scores)

            adv_scores = detector(adv_batch)
            adv_labels = torch.ones_like(adv_scores)

            loss = (
                criterion(clean_scores, clean_labels) +
                criterion(adv_scores, adv_labels)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on Validation Set 
        val_auc, _, _, _ = evaluate_detector(
            detector,
            clean_val_loader,
            adv_val_loader,
            return_y_true_scores=True
        )

        clean_scores_val = [
            detector(batch[0].to(device)).squeeze().detach().cpu().numpy()
            for batch in clean_val_loader
        ]
        adv_scores_val = [
            detector(batch[0].to(device)).squeeze().detach().cpu().numpy()
            for batch in adv_val_loader
        ]

        fpr, tpr, thresholds = roc_curve(
            np.concatenate([
                np.zeros(len(clean_val_loader.dataset)),
                np.ones(len(adv_val_loader.dataset))
            ]),
            np.concatenate(clean_scores_val + adv_scores_val)
        )

        optimal_idx = np.argmax(tpr - fpr)
        current_threshold = (
            thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
        )

        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, "
            f"Val AUC: {val_auc:.4f}, Opt Threshold: {current_threshold:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_threshold = current_threshold

    print(
        f"\nTraining complete. Best Validation AUC: {best_val_auc:.4f} "
        f"with Optimal Threshold: {best_threshold:.4f}"
    )
    return best_threshold


def evaluate_detector(
    detector,
    clean_loader,
    adv_loader,
    return_y_true_scores=False
):
    detector.eval()
    y_true, y_scores = [], []

    with torch.no_grad():
        for clean_batch, _ in clean_loader:
            scores = detector(clean_batch.to(device)).squeeze().cpu().numpy()
            y_scores.extend(scores)
            y_true.extend([0] * len(scores))

        for adv_batch, _ in adv_loader:
            scores = detector(adv_batch.to(device)).squeeze().cpu().numpy()
            y_scores.extend(scores)
            y_true.extend([1] * len(scores))

    auc = roc_auc_score(y_true, y_scores)

    if not return_y_true_scores:
        print(f"AUC Score (Detector's Insight): {auc:.4f}")

    threshold = (
        detector.rejection_threshold
        if hasattr(detector, 'rejection_threshold')
        else 0.5
    )

    return auc, np.array(y_true), np.array(y_scores), threshold


class RobustClassifier(nn.Module):
    def __init__(
        self,
        classifier,
        detector,
        feature_extractor,
        rejection_threshold=0.5
    ):
        super().__init__()
        self.classifier = classifier
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.rejection_threshold = rejection_threshold

    def forward(self, x, return_confidence=False):
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)

        detection_score = self.detector(x)
        class_output = self.classifier(features)

        should_reject = detection_score > self.rejection_threshold

        if return_confidence:
            return class_output, detection_score, should_reject
        else:
            abstain_class = torch.full_like(class_output, -1.0)
            final_output = torch.where(
                should_reject.unsqueeze(1).expand_as(class_output),
                abstain_class,
                class_output
            )
            return final_output


# ========== put this code block in a new cell ==========

# Data Preparation 

# Load CIFAR-10 Subset
transform = transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Increased subset size for more training data
subset_size = 2048

if len(dataset) < subset_size:
    print(
        f"Warning: Dataset size ({len(dataset)}) is less than "
        f"requested subset size ({subset_size}). Using full dataset."
    )
    subset_indices = list(range(len(dataset)))
else:
    subset_indices = random.sample(
        range(len(dataset)),
        subset_size
    )

subset = Subset(dataset, subset_indices)

# Split subset into training and validation for the detector
train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size

train_subset, val_subset = random_split(
    subset,
    [train_size, val_size]
)

clean_train_loader = DataLoader(
    train_subset,
    batch_size=32,
    shuffle=False
)
clean_val_loader = DataLoader(
    val_subset,
    batch_size=32,
    shuffle=False
)
clean_test_loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=False
)

# ========== put this code block in a new cell ==========

# Load Model
mobilenet = mobilenet_v2(weights="IMAGENET1K_V1")
mobilenet.classifier[1] = nn.Linear(1280, 10)
mobilenet.eval()

# Move mobilenet to device
mobilenet.to(device)

feature_extractor = nn.Sequential(
    mobilenet.features,
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
)

feature_extractor.to(device)

classifier = mobilenet.classifier

classifier.to(device)


# FGSM Attack 
def fgsm_attack(model, images, labels, epsilon=0.03):
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    return torch.clamp(perturbed, 0, 1).detach().cpu()


# ========== put this code block in a new cell ==========

# Generate Clean and Adversarial Sets 

# Detector Training
clean_train_images_list, clean_train_labels_list = [], []
adv_train_images_list, adv_train_labels_list = [], []

for images, labels in clean_train_loader:
    # Use mobilenet for attack generation
    adv_images = fgsm_attack(mobilenet, images, labels)
    clean_train_images_list.append(images.cpu())       
    clean_train_labels_list.append(labels.cpu())
    adv_train_images_list.append(adv_images.cpu())     
    adv_train_labels_list.append(labels.cpu())

clean_train_data = torch.cat(clean_train_images_list)
clean_train_labels = torch.cat(clean_train_labels_list)
adv_train_data = torch.cat(adv_train_images_list)
adv_train_labels = torch.cat(adv_train_labels_list)

train_clean_dataset = TensorDataset(
    clean_train_data,
    clean_train_labels
)
train_adv_dataset = TensorDataset(
    adv_train_data,
    adv_train_labels
)

detector_train_clean_loader = DataLoader(
    train_clean_dataset,
    batch_size=32,
    shuffle=True
)
detector_train_adv_loader = DataLoader(
    train_adv_dataset,
    batch_size=32,
    shuffle=True
)

# Detector Validation
clean_val_images_list, clean_val_labels_list = [], []
adv_val_images_list, adv_val_labels_list = [], []

for images, labels in clean_val_loader:
    adv_images = fgsm_attack(mobilenet, images, labels)
    clean_val_images_list.append(images.cpu())
    clean_val_labels_list.append(labels.cpu())
    adv_val_images_list.append(adv_images.cpu())
    adv_val_labels_list.append(labels.cpu())

clean_val_data = torch.cat(clean_val_images_list)
clean_val_labels = torch.cat(clean_val_labels_list)
adv_val_data = torch.cat(adv_val_images_list)
adv_val_labels = torch.cat(adv_val_labels_list)

val_clean_dataset = TensorDataset(
    clean_val_data,
    clean_val_labels
)
val_adv_dataset = TensorDataset(
    adv_val_data,
    adv_val_labels
)

detector_val_clean_loader = DataLoader(
    val_clean_dataset,
    batch_size=32,
    shuffle=False
)
detector_val_adv_loader = DataLoader(
    val_adv_dataset,
    batch_size=32,
    shuffle=False
)

# Final Evaluation of Robust Classifier
clean_eval_images_list, clean_eval_labels_list = [], []
adv_eval_images_list, adv_eval_labels_list = [], []

for images, labels in clean_test_loader:
    adv_images = fgsm_attack(mobilenet, images, labels)
    clean_eval_images_list.append(images.cpu())
    clean_eval_labels_list.append(labels.cpu())
    adv_eval_images_list.append(adv_images.cpu())
    adv_eval_labels_list.append(labels.cpu())

eval_clean_data = torch.cat(clean_eval_images_list)
eval_clean_labels = torch.cat(clean_eval_labels_list)
eval_adv_data = torch.cat(adv_eval_images_list)
eval_adv_labels = torch.cat(adv_eval_labels_list)

eval_clean_dataset = TensorDataset(
    eval_clean_data,
    eval_clean_labels
)
eval_adv_dataset = TensorDataset(
    eval_adv_data,
    eval_adv_labels
)

eval_clean_loader = DataLoader(
    eval_clean_dataset,
    batch_size=32,
    shuffle=False
)
eval_adv_loader = DataLoader(
    eval_adv_dataset,
    batch_size=32,
    shuffle=False
)


# ========== put this code block in a new cell ==========

# Train Detector 
detector = AdversarialDetector(feature_extractor)
detector.to(device)  

# Train the detector, get the optimal threshold
optimal_rejection_threshold = train_detector(
    detector,
    detector_train_clean_loader,
    detector_train_adv_loader,
    detector_val_clean_loader,  # Validation clean
    detector_val_adv_loader,    # Validation adversarial
    epochs=10  # Increased for better convergence
)

# Evaluate Detector (test/eval set) 
print("\n--- Final Detector Evaluation ---")
auc, y_true_eval, y_scores_eval, _ = evaluate_detector(
    detector,
    eval_clean_loader,
    eval_adv_loader,
    return_y_true_scores=True
)
print(f"Overall AUC Score for Detector: {auc:.4f}")

# Robust Classifier Test 
robust_model = RobustClassifier(
    classifier,
    detector,
    feature_extractor,
    rejection_threshold=optimal_rejection_threshold
)
robust_model.to(device)

print("\n--- Robust Classifier Performance on Clean Samples ---")
# Get a fresh batch from clean evaluation set
sample_batch_clean, sample_labels_clean = next(iter(eval_clean_loader))
sample_batch_clean = sample_batch_clean.to(device)

with torch.no_grad():
    output_clean, scores_clean, rejects_clean = robust_model(
        sample_batch_clean,
        return_confidence=True
    )
    predicted_classes_clean = torch.argmax(output_clean, dim=1)
    prediction_probs_clean = F.softmax(output_clean, dim=1)
    max_probs_clean = torch.max(prediction_probs_clean, dim=1)[0]

# Filter up to 5 accepted and 5 rejected 
accepted_clean = [
    (
        i,
        scores_clean[i].item(),
        predicted_classes_clean[i].item(),
        sample_labels_clean[i].item(),
        max_probs_clean[i].item()
    )
    for i in range(len(sample_batch_clean))
    if not rejects_clean[i].item()
][:5]

rejected_clean = [
    (
        i,
        scores_clean[i].item(),
        predicted_classes_clean[i].item(),
        sample_labels_clean[i].item(),
        max_probs_clean[i].item()
    )
    for i in range(len(sample_batch_clean))
    if rejects_clean[i].item()
][:5]

n_accept_clean = len(accepted_clean)
n_reject_clean = len(rejected_clean)

print(f"Found {n_accept_clean} accepted and "
      f"{n_reject_clean} rejected CLEAN samples in first batch.")


# ========== put this code block in a new cell ==========

# Visualization
fig_clean, axs_clean = plt.subplots(
    2, max(n_accept_clean, n_reject_clean, 1),
    figsize=(3 * max(n_accept_clean, n_reject_clean, 1), 6)
)
fig_clean.suptitle(
    f"Robust Classifier on CLEAN Samples (Threshold: {optimal_rejection_threshold:.3f})\n"
    "Top Row: Accepted | Bottom Row: Rejected", fontsize=14
)

# Normalize axis shape
if max(n_accept_clean, n_reject_clean, 1) == 1:
    axs_clean = axs_clean.reshape(-1, 1)
elif len(axs_clean.shape) == 1:
    axs_clean = axs_clean.reshape(-1, 1)

# Accepted clean (green)
for col, (i, det_score, pred, true, conf) in enumerate(accepted_clean):
    ax = axs_clean[0, col]
    img = sample_batch_clean[i].permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype('uint8')
    ax.imshow(img)
    ax.set_title(
        f"{classes[pred]} (conf: {conf:.2f})\n"
        f"Det: {det_score:.3f} | True: {classes[true]}", fontsize=9
    )
    ax.axis('off')
    rect = patches.Rectangle(
        (0, 0), img.shape[1]-1, img.shape[0]-1,
        linewidth=3, edgecolor='green', facecolor='none'
    )
    ax.add_patch(rect)

# Rejected clean (red)
for col, (i, det_score, pred, true, conf) in enumerate(rejected_clean):
    ax = axs_clean[1, col]
    img = sample_batch_clean[i].permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype('uint8')
    ax.imshow(img)
    ax.set_title(
        f"{classes[pred]} (conf: {conf:.2f})\n"
        f"Det: {det_score:.3f} | True: {classes[true]}", fontsize=9
    )
    ax.axis('off')
    rect = patches.Rectangle(
        (0, 0), img.shape[1]-1, img.shape[0]-1,
        linewidth=3, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

# Turn off empty slots
max_cols_clean = max(n_accept_clean, n_reject_clean, 1)
for row in range(2):
    for col in range(max_cols_clean):
        if (row == 0 and col >= n_accept_clean) or (row == 1 and col >= n_reject_clean):
            axs_clean[row, col].axis('off')
            axs_clean[row, col].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

# Adversarial Visualization 
print("\n--- Robust Classifier Performance on ADVERSARIAL Samples ---")

sample_batch_adv, sample_labels_adv = next(iter(eval_adv_loader))
sample_batch_adv = sample_batch_adv.to(device)

with torch.no_grad():
    output_adv, scores_adv, rejects_adv = robust_model(
        sample_batch_adv, return_confidence=True
    )
    predicted_classes_adv = torch.argmax(output_adv, dim=1)
    prediction_probs_adv = F.softmax(output_adv, dim=1)
    max_probs_adv = torch.max(prediction_probs_adv, dim=1)[0]

accepted_adv = [
    (i, scores_adv[i].item(), predicted_classes_adv[i].item(),
     sample_labels_adv[i].item(), max_probs_adv[i].item())
    for i in range(len(sample_batch_adv)) if not rejects_adv[i].item()
][:5]

rejected_adv = [
    (i, scores_adv[i].item(), predicted_classes_adv[i].item(),
     sample_labels_adv[i].item(), max_probs_adv[i].item())
    for i in range(len(sample_batch_adv)) if rejects_adv[i].item()
][:5]

n_accept_adv = len(accepted_adv)
n_reject_adv = len(rejected_adv)

print(f"Found {n_accept_adv} accepted and {n_reject_adv} rejected ADVERSARIAL samples.")

fig_adv, axs_adv = plt.subplots(
    2, max(n_accept_adv, n_reject_adv, 1),
    figsize=(3 * max(n_accept_adv, n_reject_adv, 1), 6)
)
fig_adv.suptitle(
    f"Robust Classifier on ADVERSARIAL Samples (Threshold: {optimal_rejection_threshold:.3f})\n"
    "Top Row: Accepted | Bottom Row: Rejected", fontsize=14
)

if max(n_accept_adv, n_reject_adv, 1) == 1:
    axs_adv = axs_adv.reshape(-1, 1)
elif len(axs_adv.shape) == 1:
    axs_adv = axs_adv.reshape(-1, 1)

# Accepted adversarial (green)
for col, (i, det_score, pred, true, conf) in enumerate(accepted_adv):
    ax = axs_adv[0, col]
    img = sample_batch_adv[i].permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype('uint8')
    ax.imshow(img)
    ax.set_title(
        f"{classes[pred]} (conf: {conf:.2f})\n"
        f"Det: {det_score:.3f} | True: {classes[true]}", fontsize=9
    )
    ax.axis('off')
    rect = patches.Rectangle(
        (0, 0), img.shape[1]-1, img.shape[0]-1,
        linewidth=3, edgecolor='green', facecolor='none'
    )
    ax.add_patch(rect)

# Rejected adversarial (red)
for col, (i, det_score, pred, true, conf) in enumerate(rejected_adv):
    ax = axs_adv[1, col]
    img = sample_batch_adv[i].permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype('uint8')
    ax.imshow(img)
    ax.set_title(
        f"{classes[pred]} (conf: {conf:.2f})\n"
        f"Det: {det_score:.3f} | True: {classes[true]}", fontsize=9
    )
    ax.axis('off')
    rect = patches.Rectangle(
        (0, 0), img.shape[1]-1, img.shape[0]-1,
        linewidth=3, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

max_cols_adv = max(n_accept_adv, n_reject_adv, 1)
for row in range(2):
    for col in range(max_cols_adv):
        if (row == 0 and col >= n_accept_adv) or (row == 1 and col >= n_reject_adv):
            axs_adv[row, col].axis('off')
            axs_adv[row, col].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

# ========== put this code block in a new cell ==========

# Detailed Analysis for the entire eval set 
print("\n" + "=" * 60)
print("COMPREHENSIVE ROBUST CLASSIFIER ANALYSIS")
print("=" * 60)

# Clean Set Evaluation
total_clean_samples = 0
correctly_classified_clean_accepted = 0
incorrectly_classified_clean_accepted = 0
rejected_clean_count = 0

for images, labels in eval_clean_loader:
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        output, scores, rejects = robust_model(
            images, return_confidence=True
        )
        predicted_classes = torch.argmax(output, dim=1)
    
    total_clean_samples += len(images)
    rejected_clean_count += rejects.sum().item()
    
    for i in range(len(images)):
        if not rejects[i]:
            if predicted_classes[i] == labels[i]:
                correctly_classified_clean_accepted += 1
            else:
                incorrectly_classified_clean_accepted += 1

accepted_clean_count = total_clean_samples - rejected_clean_count
clean_acceptance_rate = accepted_clean_count / total_clean_samples

print(f"\n--- Clean Samples (Total: {total_clean_samples}) ---")
print(f"  Accepted: {accepted_clean_count} "
      f"({clean_acceptance_rate:.1%})")
print(f"  Rejected: {rejected_clean_count} "
      f"({(1 - clean_acceptance_rate):.1%})")

if accepted_clean_count > 0:
    clean_accuracy_on_accepted = (
        correctly_classified_clean_accepted / accepted_clean_count
    )
    print(f"  Accuracy on Accepted Clean Samples: "
          f"{clean_accuracy_on_accepted:.2%}")
else:
    print("  No clean samples were accepted for classification.")

# Adversarial Set Evaluation 
total_adv_samples = 0
correctly_classified_adv_accepted = 0
incorrectly_classified_adv_accepted = 0
rejected_adv_count = 0

for images, labels in eval_adv_loader:
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        output, scores, rejects = robust_model(
            images, return_confidence=True
        )
        predicted_classes = torch.argmax(output, dim=1)
    
    total_adv_samples += len(images)
    rejected_adv_count += rejects.sum().item()
    
    for i in range(len(images)):
        if not rejects[i]:
            if predicted_classes[i] == labels[i]:
                correctly_classified_adv_accepted += 1
            else:
                incorrectly_classified_adv_accepted += 1

accepted_adv_count = total_adv_samples - rejected_adv_count
adv_acceptance_rate = accepted_adv_count / total_adv_samples

print(f"\n--- Adversarial Samples (Total: {total_adv_samples}) ---")
print(f"  Accepted: {accepted_adv_count} "
      f"({adv_acceptance_rate:.1%})")
print(f"  Rejected: {rejected_adv_count} "
      f"({(1 - adv_acceptance_rate):.1%})")

if accepted_adv_count > 0:
    adv_accuracy_on_accepted = (
        correctly_classified_adv_accepted / accepted_adv_count
    )
    print(f"  Accuracy on Accepted Adversarial Samples: "
          f"{adv_accuracy_on_accepted:.2%}")
else:
    print("  No adversarial samples were accepted for classification.")

# Summary
print("\n--- Overall Summary ---")
print(f"  Optimal Rejection Threshold: "
      f"{optimal_rejection_threshold:.4f}")
print(f"  Clean Acceptance Rate: {clean_acceptance_rate:.1%}")
print(f"  Adversarial Rejection Rate (True Positives): "
      f"{(1 - adv_acceptance_rate):.1%}")
print(f"  Clean Rejection Rate (False Positives): "
      f"{(1 - clean_acceptance_rate):.1%}")

detection_effective = (
    (1 - adv_acceptance_rate) > (1 - clean_acceptance_rate) and
    (1 - adv_acceptance_rate) > 0.5
)

print(f"\n  Detection Effectiveness: "
      f"{'Effective' if detection_effective else 'Needs Improvement'}")
