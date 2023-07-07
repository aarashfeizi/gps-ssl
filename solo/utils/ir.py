from typing import Tuple

import torch
import torch.nn.functional as F
from torchmetrics.metric import Metric
import numpy as np
import copy

class ImageRetrieval(Metric):
    def __init__(
        self,
        k: int = 20,
        max_distance_matrix_size: int = int(5e6),
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
        dist_sync_on_step: bool = False,
        ratios=[0.5],
    ):
        """Implements the weighted k-NN classifier used for evaluation.

        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            max_distance_matrix_size (int, optional): maximum number of elements in the
                distance matrix. Defaults to 5e6.
            distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
                "euclidean". Defaults to "cosine".
            epsilon (float, optional): Small value for numerical stability. Only used with
                euclidean distance. Defaults to 0.00001.
            dist_sync_on_step (bool, optional): whether to sync distributed values at every
                step. Defaults to False.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.k = k
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon
        self.ratios = ratios

        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def update(
        self,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.

        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (test_features is None) == (test_targets is None)

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    @torch.no_grad()
    def compute(self) -> Tuple[float]:
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """

        test_features = torch.cat(self.test_features)
        test_targets = torch.cat(self.test_targets)

        if self.distance_fx == "cosine":
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_test_images = test_targets.size(0)
        
        # chunk_size = min(
        #     max(1, self.max_distance_matrix_size // num_train_images),
        #         num_test_images)
        
        chunk_size = num_test_images
        
        # k = min(self.k, num_train_images)
        k = min(self.k + 1, num_test_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(test_features.device)
        for idx in range(0, num_test_images, chunk_size):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarities = torch.mm(features, test_features.t())
            elif self.distance_fx == "euclidean":
                similarities = 1 / (torch.cdist(features, test_features) + self.epsilon)
            else:
                raise NotImplementedError
            
            similarities, indices = similarities.topk(k + 1, largest=True, sorted=True)
            similarities = similarities[:, 1:]
            indices = indices[:, 1:]
            candidates = test_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)
            
            for i in range(len(targets)):
                t = targets[i]
                if t in retrieved_neighbors[i, :1]:
                    top1 += 1
                if t in retrieved_neighbors[i, :5]:
                    top5 += 1
            total += targets.size(0)
            
        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        return top1, top5

    @torch.no_grad()
    def compute_ur(self) -> Tuple[float]:
        """Computes weighted underrepresented accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """

        all_test_features = torch.cat(self.test_features)
        all_test_targets = torch.cat(self.test_targets)
        output = {}

        for ratio in self.ratios:
            percentile = np.percentile(all_test_targets, ratio * 100)
            test_features = all_test_features[all_test_targets < percentile]
            test_targets = all_test_targets[all_test_targets < percentile]
            if self.distance_fx == "cosine":
                test_features = F.normalize(test_features)

            num_classes = torch.unique(test_targets).numel()
            num_test_images = test_targets.size(0)
            
            # chunk_size = min(
            #     max(1, self.max_distance_matrix_size // num_train_images),
            #         num_test_images)
            
            chunk_size = num_test_images
            
            # k = min(self.k, num_train_images)
            k = min(self.k + 1, num_test_images)

            top1, top5, total = 0.0, 0.0, 0
            retrieval_one_hot = torch.zeros(k, num_classes).to(test_features.device)
            for idx in range(0, num_test_images, chunk_size):
                # get the features for test images
                features = test_features[idx : min((idx + chunk_size), num_test_images), :]
                targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
                batch_size = targets.size(0)

                # calculate the dot product and compute top-k neighbors
                if self.distance_fx == "cosine":
                    similarities = torch.mm(features, test_features.t())
                elif self.distance_fx == "euclidean":
                    similarities = 1 / (torch.cdist(features, test_features) + self.epsilon)
                else:
                    raise NotImplementedError
                
                similarities, indices = similarities.topk(k + 1, largest=True, sorted=True)
                similarities = similarities[:, 1:]
                indices = indices[:, 1:]
                candidates = test_targets.view(1, -1).expand(batch_size, -1)
                retrieved_neighbors = torch.gather(candidates, 1, indices)
                
                for i in range(len(targets)):
                    t = targets[i]
                    if t in retrieved_neighbors[i, :1]:
                        top1 += 1
                    if t in retrieved_neighbors[i, :5]:
                        top5 += 1
                total += targets.size(0)
                
            top1 = top1 * 100.0 / total
            top5 = top5 * 100.0 / total

            output[ratio] = {'top1': top1, 'top5': top5}

        return output


    def make_batch_bce_labels(self, labels, diagonal_fill=None):
        """
        :param labels: e.g. tensor of size (N,1)
        :return: binary matrix of labels of size (N, N)
        """

        l_ = labels.repeat(len(labels)).reshape(-1, len(labels))
        l__ = labels.repeat_interleave(len(labels)).reshape(-1, len(labels))

        final_bce_labels = (l_ == l__).type(torch.float32)

        if diagonal_fill:
            final_bce_labels.fill_diagonal_(diagonal_fill)

        return final_bce_labels


    def get_xs_ys(self, target_labels, k=1, bce_labels=None, pairwise_hard_neg=True):
        """

        :param target_labels: tensor of (N, N) with 0s and 1s
        :param k: number of pos and neg samples per anch
        :param bce_labels: tensor of (N, N) with 0s and 1s (without considering pairwise labels)
        :return: an equal number of positive and negative pairs chosen randomly

        """
        xs = []
        ys = []
        target_labels_copy = copy.deepcopy(target_labels)
        target_labels_copy.fill_diagonal_(-1)
        bce_labels_copy = None
        if bce_labels is not None:
            bce_labels_copy = copy.deepcopy(bce_labels)
            bce_labels_copy.fill_diagonal_(-1)
        for i, row in enumerate(target_labels_copy):

            neg_idx = torch.where(row == 0)[0]
            pos_idx = torch.where(row == 1)[0]

            if len(pos_idx) == 0:
                print(f'skipping {i}... not enough positive samples')
                continue


            challenging_neg_idx = None
            if bce_labels_copy is not None and pairwise_hard_neg:
                bce_row = bce_labels_copy[i, :]
                challenging_negatives = bce_row - row
                challenging_neg_idx = torch.where(challenging_negatives == 1)[0]

            ys.extend(self.get_samples(neg_idx, k))

            if challenging_neg_idx is None or \
                    len(challenging_neg_idx) == 0:
                # if len(challenging_neg_idx) == 0:
                    # print('couldnt do challenging, 0!!')
                ys.extend(self.get_samples(neg_idx, k))
            else:
                ys.extend(self.get_samples(challenging_neg_idx, k))

            ys.extend(self.get_samples(pos_idx, 2 * k))
            xs.extend(self.get_samples([i], 4 * k))

        return xs, ys


    def get_samples(self, l, k):
        if len(l) < k:
            to_ret = np.random.choice(l, k, replace=True)
        else:
            to_ret = np.random.choice(l, k, replace=False)
        return to_ret

    def get_hard_xs_ys(self, bce_labels, a2n, k):
        """

        :param bce_labels: tensor of (N, N) with 0s and 1s
        :param a2n: dict, mapping every anchor idx to hard neg idxs
        :param k: number of pos and neg samples per anch
        :return:

        """
        xs = []
        ys = []
        bce_labels_copy = copy.deepcopy(bce_labels)
        bce_labels_copy.fill_diagonal_(-1)
        for i, row in enumerate(bce_labels_copy):
            neg_idx_chosen = a2n[i][:k]
            pos_idx = torch.where(row == 1)[0]

            ys.extend(neg_idx_chosen)
            ys.extend(self.get_samples(pos_idx, k))
            xs.extend(self.get_samples([i], 2 * k))

        return xs, ys


    def compute_auroc(self, k=1, pairwise_labels=None, pairwise_hard_neg=True):
        """

        :param pairwise_labels: a (N, N) binary matrix with pairwise labels
        :return: the AUROC score, where random would score 1/(k + 1)
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics.pairwise import cosine_similarity

        test_features = torch.cat(self.test_features).cpu()
        test_targets = torch.cat(self.test_targets).cpu()

        bce_labels = self.make_batch_bce_labels(test_targets)

        if pairwise_labels is None:
            pairwise = False
            target_labels = bce_labels
        else:
            pairwise = True
            target_labels = pairwise_labels

        if target_labels.dtype != torch.float32:
            target_labels = target_labels.type(torch.float32)

        similarities = cosine_similarity(test_features)
        
        if pairwise:
            xs, ys = self.get_xs_ys(target_labels, k=k, bce_labels=bce_labels, pairwise_hard_neg=pairwise_hard_neg)
        else:
            xs, ys = self.get_xs_ys(target_labels, k=k, bce_labels=None)

        true_labels = target_labels[xs, ys]
        predicted_labels = similarities[xs, ys]

        return roc_auc_score(true_labels, predicted_labels), {'true_labels': true_labels,
                                                            'pred_labels': predicted_labels}


