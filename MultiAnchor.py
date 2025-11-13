import torch
import torch.nn as nn
from pytorch_metric_learning import distances, reducers
from sklearn.neighbors import BallTree
import numpy as np

# Define MultiAnchorLoss class
class MultiAnchorLoss(nn.Module):
    def __init__(self, margin1=1.0, margin2=0.5, margin3=0.5, lambda_reg=0.01):
        """
        Multi-anchor loss function.
        """
        super(MultiAnchorLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.margin3 = margin3
        self.lambda_reg = lambda_reg
        self.distance = distances.LpDistance(p=2)
        self.reducer = reducers.AvgNonZeroReducer()

    def forward(self, embeddings, labels, mining_results):
        device = embeddings.device

        #  Compute loss components
        loss1 = self.compute_loss_part(
            embeddings,
            mining_results.get("positive_pairs", torch.empty(0, 2, dtype=torch.long, device=device)),
            mining_results.get("negative_pairs", torch.empty(0, 2, dtype=torch.long, device=device)),
            self.margin1
        )
        loss2 = self.compute_overlap_loss(
            embeddings,
            mining_results.get("overlap_positive_pairs", torch.empty(0, 2, dtype=torch.long, device=device)),
            self.margin2
        )
        loss3 = self.compute_overlap_loss(
            embeddings,
            mining_results.get("overlap_negative_pairs", torch.empty(0, 2, dtype=torch.long, device=device)),
            self.margin3
        )
        loss4 = self.compute_triplet_loss(
            embeddings,
            mining_results.get("triplets", torch.empty(0, 3, dtype=torch.long, device=device)),
            self.margin2, self.margin3
        )

        loss_list = []
        if isinstance(loss1, torch.Tensor) and loss1.numel() > 0:
            loss_list.append(loss1)
        if isinstance(loss2, torch.Tensor) and loss2.numel() > 0:
            loss_list.append(loss2)
        if isinstance(loss3, torch.Tensor) and loss3.numel() > 0:
            loss_list.append(loss3)
        if isinstance(loss4, torch.Tensor) and loss4.numel() > 0:
            loss_list.append(loss4)

        if len(loss_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        all_losses = torch.cat(loss_list, dim=0)
        indices = torch.arange(len(all_losses), device=device)

        loss_dict = {
            "multi_anchor_loss": {
                "losses": all_losses,
                "indices": indices,
                "reduction_type": "element"
            }
        }

        loss = self.reducer(loss_dict, embeddings, labels)

        reg_term = self.lambda_reg * self.get_regularization_term()
        loss = loss + reg_term

        return loss

    def compute_loss_part(self, embeddings, positive_pairs, negative_pairs, margin):
        #  Compute basic contrastive loss
        device = embeddings.device
        if positive_pairs.size(0) == 0 or negative_pairs.size(0) == 0:
            return torch.tensor([], device=device)

        pos_distances = self.distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]])
        neg_distances = self.distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])

        if pos_distances.dim() == 0:
            pos_distances = pos_distances.unsqueeze(0)
        if neg_distances.dim() == 0:
            neg_distances = neg_distances.unsqueeze(0)

        losses = torch.relu(pos_distances - neg_distances + margin)

        if losses.dim() == 0:
            losses = losses.unsqueeze(0)

        return losses

    def compute_overlap_loss(self, embeddings, pairs, margin):
        device = embeddings.device
        if pairs.size(0) == 0:
            return torch.tensor([], device=device)

        indices_0 = pairs[:, 0]
        indices_1 = pairs[:, 1]
        if indices_0.numel() == 0 or indices_1.numel() == 0:
            return torch.tensor([], device=device)

        anchor_distances = self.distance(embeddings[indices_0], embeddings[indices_1])

        if anchor_distances.dim() == 0:
            anchor_distances = anchor_distances.unsqueeze(0)

        losses = torch.relu(-anchor_distances + margin)

        if losses.dim() == 0:
            losses = losses.unsqueeze(0)

        return losses

    def compute_triplet_loss(self, embeddings, triplets, margin2, margin3):
        device = embeddings.device
        if triplets.size(0) == 0:
            return torch.tensor([], device=device)

        anchor = embeddings[triplets[:, 0]]
        positive = embeddings[triplets[:, 1]]
        negative = embeddings[triplets[:, 2]]

        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
            positive = positive.unsqueeze(0)
            negative = negative.unsqueeze(0)

        pos_distances = self.distance(anchor, positive)
        neg_distances = self.distance(anchor, negative)

        if pos_distances.dim() == 0:
            pos_distances = pos_distances.unsqueeze(0)
        if neg_distances.dim() == 0:
            neg_distances = neg_distances.unsqueeze(0)

        losses = torch.relu(pos_distances - neg_distances + margin2 + margin3)

        if losses.dim() == 0:
            losses = losses.unsqueeze(0)

        return losses

    def get_regularization_term(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss = reg_loss + torch.norm(param)
        return reg_loss

# Define MultiAnchorMiner class
class MultiAnchorMiner:
    def __init__(self, num_hard_samples=None, k=7):
        """
        Multi-anchor miner.
        """
        self.num_hard_samples = num_hard_samples
        self.k = k

    def __call__(self, embeddings, labels):
        """
        Mine pairs and triplets.
        """
        device = embeddings.device

        #  Get lower approximation and overlap indices using rough_set
        lower_approx_idx, overlap_idx = rough_set(embeddings, labels, self.k)

        embeddings = embeddings.to(device)
        labels = labels.to(device)

        positive_pairs = []
        negative_pairs = []
        overlap_positive_pairs = []
        overlap_negative_pairs = []
        triplets = []

        with torch.no_grad():
            distances_matrix = torch.cdist(embeddings, embeddings, p=2)

        #  Positive/negative pairs in lower approximation set
        pos_pairs, neg_pairs = self.get_pairs(lower_approx_idx, labels, distances_matrix)
        positive_pairs.extend(pos_pairs)
        negative_pairs.extend(neg_pairs)

        #  Cross-pairs between lower approx. and overlap
        overlap_pos_pairs, overlap_neg_pairs = self.get_cross_pairs(
            lower_approx_idx, labels, overlap_idx, labels, distances_matrix
        )
        overlap_positive_pairs.extend(overlap_pos_pairs)
        overlap_negative_pairs.extend(overlap_neg_pairs)

        #  Triplets within the overlap region
        triplets.extend(self.get_triplets(overlap_idx, labels, distances_matrix))

        mining_results = {
            "positive_pairs": torch.tensor(positive_pairs, device=device, dtype=torch.long) if positive_pairs else torch.empty(0, 2, dtype=torch.long, device=device),
            "negative_pairs": torch.tensor(negative_pairs, device=device, dtype=torch.long) if negative_pairs else torch.empty(0, 2, dtype=torch.long, device=device),
            "overlap_positive_pairs": torch.tensor(overlap_positive_pairs, device=device, dtype=torch.long) if overlap_positive_pairs else torch.empty(0, 2, dtype=torch.long, device=device),
            "overlap_negative_pairs": torch.tensor(overlap_negative_pairs, device=device, dtype=torch.long) if overlap_negative_pairs else torch.empty(0, 2, dtype=torch.long, device=device),
            "triplets": torch.tensor(triplets, device=device, dtype=torch.long) if triplets else torch.empty(0, 3, dtype=torch.long, device=device),
        }

        return mining_results

    def get_pairs(self, idxs, labels, distances):
        pos_pairs = []
        neg_pairs = []

        idxs = idxs.tolist() if isinstance(idxs, torch.Tensor) else idxs

        for i in range(len(idxs)):
            idx_i = idxs[i]
            label_i = labels[idx_i].item()
            for j in range(i + 1, len(idxs)):
                idx_j = idxs[j]
                label_j = labels[idx_j].item()
                if label_i == label_j:
                    distance = distances[idx_i, idx_j].item()
                    pos_pairs.append([idx_i, idx_j, distance])
                else:
                    distance = distances[idx_i, idx_j].item()
                    neg_pairs.append([idx_i, idx_j, distance])

        pos_pairs = self.mine_hard_samples(pos_pairs, hardest=True)
        neg_pairs = self.mine_hard_samples(neg_pairs, hardest=False)

        pos_pairs = [[p[0], p[1]] for p in pos_pairs]
        neg_pairs = [[p[0], p[1]] for p in neg_pairs]

        return pos_pairs, neg_pairs

    def get_cross_pairs(self, idxs1, labels1, idxs2, labels2, distances):
        pos_pairs = []
        neg_pairs = []

        idxs1 = idxs1.tolist() if isinstance(idxs1, torch.Tensor) else idxs1
        idxs2 = idxs2.tolist() if isinstance(idxs2, torch.Tensor) else idxs2

        for idx_i in idxs1:
            label_i = labels1[idx_i].item()
            for idx_j in idxs2:
                label_j = labels2[idx_j].item()
                if label_i == label_j:
                    distance = distances[idx_i, idx_j].item()
                    pos_pairs.append([idx_i, idx_j, distance])
                else:
                    distance = distances[idx_i, idx_j].item()
                    neg_pairs.append([idx_i, idx_j, distance])

        pos_pairs = self.mine_hard_samples(pos_pairs, hardest=True)
        neg_pairs = self.mine_hard_samples(neg_pairs, hardest=False)

        pos_pairs = [[p[0], p[1]] for p in pos_pairs]
        neg_pairs = [[p[0], p[1]] for p in neg_pairs]

        return pos_pairs, neg_pairs

    def get_triplets(self, idxs, labels, distances):
        triplets = []
        label_to_idxs = {}

        idxs = idxs.tolist() if isinstance(idxs, torch.Tensor) else idxs

        for idx in idxs:
            label = labels[idx].item()
            if label not in label_to_idxs:
                label_to_idxs[label] = []
            label_to_idxs[label].append(idx)

        for label in label_to_idxs:
            pos_idxs = label_to_idxs[label]
            neg_labels = [l for l in label_to_idxs if l != label]
            neg_idxs = [idx for l in neg_labels for idx in label_to_idxs[l]]

            for anchor_idx in pos_idxs:
                positive_idxs = [idx for idx in pos_idxs if idx != anchor_idx]
                if not positive_idxs or not neg_idxs:
                    continue

                pos_distances = [(idx, distances[anchor_idx, idx].item()) for idx in positive_idxs]
                pos_distances = sorted(pos_distances, key=lambda x: x[1], reverse=True)

                neg_distances = [(idx, distances[anchor_idx, idx].item()) for idx in neg_idxs]
                neg_distances = sorted(neg_distances, key=lambda x: x[1])

                num_samples = self.num_hard_samples if self.num_hard_samples else min(len(pos_distances), len(neg_distances))

                for i in range(num_samples):
                    pos_idx = pos_distances[i % len(pos_distances)][0]
                    neg_idx = neg_distances[i % len(neg_distances)][0]
                    triplets.append([anchor_idx, pos_idx, neg_idx])

        return triplets

    def mine_hard_samples(self, pairs, hardest=True):
        if not pairs:
            return []

        pairs = sorted(pairs, key=lambda x: x[2], reverse=hardest)

        if self.num_hard_samples:
            pairs = pairs[:self.num_hard_samples]

        return pairs

# Define rough_set function
def rough_set(embeddings, labels, k=5):
    """
    Return lower approximation and overlap indices based on embeddings and labels.
    """
    embeddings_np = embeddings.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()
    n_samples = embeddings_np.shape[0]

    tree = BallTree(embeddings_np, metric='euclidean')

    N_minus_C = {}
    N_plus_C = {}
    Bn_C = {}
    classes = np.unique(labels_np)

    for cls in classes:
        N_minus_C[cls] = set()
        N_plus_C[cls] = set()

    for i in range(n_samples):
        #Find k-nearest neighbors
        dist, ind = tree.query([embeddings_np[i]], k=k)
        neighbor_classes = labels_np[ind[0]]

        current_class = labels_np[i]

        if np.all(neighbor_classes == current_class):
            N_minus_C[current_class].add(i)
        if np.any(neighbor_classes == current_class):
            N_plus_C[current_class].add(i)

    for cls in classes:
        N_minus_set = N_minus_C[cls]
        N_plus_set = N_plus_C[cls]
        Bn_C[cls] = N_plus_set - N_minus_set

    overlap_region = set()
    class_list = list(classes)
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            cls_i = class_list[i]
            cls_j = class_list[j]
            overlap = Bn_C[cls_i] & Bn_C[cls_j]
            overlap_region.update(overlap)

    lower_approximation = set()
    for cls in classes:
        lower_approximation.update(N_minus_C[cls])

    overlap_region_idx = torch.tensor(list(overlap_region), dtype=torch.long, device=embeddings.device)
    lower_approximation_idx = torch.tensor(list(lower_approximation), dtype=torch.long, device=embeddings.device)

    return lower_approximation_idx, overlap_region_idx