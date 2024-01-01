# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from solo.data.h5_dataset import H5Dataset
from timm.models.helpers import group_parameters
from timm.optim.optim_factory import _layer_map
from tqdm import tqdm
from PIL import Image
import faiss
from sklearn import metrics
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

MAP_STRING_TO_BOOL = {'true': True,
                      'false': False}


def _1d_filter(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.isfinite()


def _2d_filter(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.isfinite().all(dim=1)


def _single_input_filter(tensor: torch.Tensor) -> Tuple[torch.Tensor]:
    if len(tensor.size()) == 1:
        filter_func = _1d_filter
    elif len(tensor.size()) == 2:
        filter_func = _2d_filter
    else:
        raise RuntimeError("Only 1d and 2d tensors are supported.")

    selected = filter_func(tensor)
    tensor = tensor[selected]

    return tensor, selected


def _multi_input_filter(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor]:
    if len(tensors[0].size()) == 1:
        filter_func = _1d_filter
    elif len(tensors[0].size()) == 2:
        filter_func = _2d_filter
    else:
        raise RuntimeError("Only 1d and 2d tensors are supported.")

    selected = filter_func(tensors[0])
    for tensor in tensors[1:]:
        selected = torch.logical_and(selected, filter_func(tensor))
    tensors = [tensor[selected] for tensor in tensors]

    return tensors, selected


def filter_inf_n_nan(tensors: List[torch.Tensor], return_indexes: bool = False):
    """Filters out inf and nans from any tensor.
    This is usefull when there are instability issues,
    which cause a small number of values to go bad.

    Args:
        tensor (List): tensor to remove nans and infs from.

    Returns:
        torch.Tensor: filtered view of the tensor without nans or infs.
    """

    if isinstance(tensors, torch.Tensor):
        tensors, selected = _single_input_filter(tensors)
    else:
        tensors, selected = _multi_input_filter(tensors)

    if return_indexes:
        return tensors, selected
    return tensors


class FilterInfNNan(nn.Module):
    def __init__(self, module):
        """Layer that filters out inf and nans from any tensor.
        This is usefull when there are instability issues,
        which cause a small number of values to go bad.

        Args:
            tensor (List): tensor to remove nans and infs from.

        Returns:
            torch.Tensor: filtered view of the tensor without nans or infs.
        """
        super().__init__()

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        out = filter_inf_n_nan(out)
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "module":
                raise AttributeError()
            return getattr(self.module, name)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Copy & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    def norm_cdf(x):
        """Computes standard normal cumulative distribution function"""

        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        logging.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Copy & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


@torch.no_grad()
def concat_all_gather_no_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    if dist.is_available() and dist.is_initialized():
        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    return tensor


def compute_dataset_size(
    dataset: Optional[str] = None,
    train: Optional[bool] = True,
    data_path: Optional[str] = None,
    data_format: Optional[str] = "image_folder",
    no_labels: Optional[bool] = False,
    data_fraction: Optional[float] = -1,
):
    """Utility function to get the dataset size. If using cifar or stl,
    provide dataset and the train flag.
    E.g., compute_dataset_size(dataset='cifar10', train=True/False).
    When using an ImageFolder dataset, just provide the path to the folder and
    specify if it has labels or not with the no_labels flag.

    Args:
        dataset (Optional[str]): dataset size for predefined datasets
            [cifar10, cifar100, stl10]. Defaults to None.
        train (Optional[bool]): train dataset flag. Defaults to True.
        data_path (Optional[str]): path to the folder. Defaults to None.
        data_format (Optional[str]): format of the data, either "image_folder" or "h5".
            Defaults to "image_folder".
        no_labels (Optional[bool]): if the dataset has no labels. Defaults to False.
        data_fraction (Optional[float]): amount of data to use. Defaults to -1.

    Returns:
        int: size of the dataset
    """

    DATASET_SIZES = {
        "cifar10": {"train": 50_000, "val": 10_000},
        "cifar100": {"train": 50_000, "val": 10_000},
        "stl10": {"train": 105_000, "val": 8_000},
    }
    size = None

    if dataset is not None:
        size = DATASET_SIZES.get(dataset.lower(), {}).get("train" if train else "val", None)

    if data_format == "h5":
        size = len(H5Dataset(dataset, data_path))

    if size is None:
        if no_labels:
            size = len(os.listdir(data_path))
        else:
            size = sum(
                len(os.listdir(os.path.join(data_path, class_))) for class_ in os.listdir(data_path)
            )

    if data_fraction != -1:
        size = int(size * data_fraction)

    return size


def make_contiguous(module):
    """Make the model contigous in order to comply with some distributed strategies.
    https://github.com/lucidrains/DALLE-pytorch/issues/330
    """

    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())


def generate_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Adapted from https://github.com/facebookresearch/mae.
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = generate_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def generate_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # Adapted from https://github.com/facebookresearch/mae.

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = generate_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = generate_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def generate_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Adapted from https://github.com/facebookresearch/mae.
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def param_groups_layer_decay(
    model: nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: Tuple[str] = (),
    layer_decay: float = 0.75,
):
    """
    Parameter groups for layer-wise lr decay & weight decay
    Based on BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """

    no_weight_decay_list = set(no_weight_decay_list)
    param_group_names = {}  # NOTE for debugging
    param_groups = {}

    if hasattr(model, "group_matcher"):
        # FIXME interface needs more work
        layer_map = group_parameters(model, model.group_matcher(coarse=False), reverse=True)
    else:
        # fallback
        layer_map = _layer_map(model)
    num_layers = max(layer_map.values()) + 1
    layer_max = num_layers - 1
    layer_scales = list(layer_decay ** (layer_max - i) for i in range(num_layers))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = layer_map.get(name, layer_max)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "param_names": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())


def remove_bias_and_norm_from_weight_decay(parameter_groups: List[Dict]):
    out = []
    for group in parameter_groups:
        # parameters with weight decay
        decay_group = {k: v for k, v in group.items() if k != "params"}

        # parameters without weight decay
        no_decay_group = {k: v for k, v in group.items() if k != "params"}
        no_decay_group["weight_decay"] = 0
        group_name = group.get("name", None)
        if group_name:
            no_decay_group["name"] = group_name + "_no_decay"

        # split parameters into the two lists
        decay_params = []
        no_decay_params = []
        for param in group["params"]:
            if param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # add groups back
        if decay_params:
            decay_group["params"] = decay_params
            out.append(decay_group)
        if no_decay_params:
            no_decay_group["params"] = no_decay_params
            out.append(no_decay_group)
    return out


def omegaconf_select(cfg, key, default=None):
    """Wrapper for OmegaConf.select to allow None to be returned instead of 'None'."""
    value = OmegaConf.select(cfg, key, default=default)
    if value == "None":
        return None
    return value

def train_emb_model(cfg, model, train_loader, val_loader=None, supervised=False):
    import wandb
    OPTS = {'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW}
    
    LOSSES = {'mse': torch.nn.MSELoss,
                'ce': torch.nn.CrossEntropyLoss}

    epochs = cfg.emb_model.epochs
    loss_fn = LOSSES[cfg.emb_model.loss]()
    optimizer = OPTS[cfg.emb_model.opt](model.parameters(),
                                lr=cfg.emb_model.lr,
                                weight_decay=cfg.emb_model.weight_decay)

    def train_one_step(current_epoch):
        total_loss = 0
        pred_lbls = []
        true_lbls = []
        with tqdm(total=len(train_loader), desc=f'{current_epoch}/{epochs}') as t:
            for idx, batch in enumerate(train_loader, start=1):
                _, X, true_lbl = batch
                X = X.cuda()
                X_pred = model(X)

                acc = None
                if supervised:
                    true_lbl = true_lbl.cuda()
                    loss = loss_fn(X_pred, true_lbl)
                    pred_lbls.extend(X_pred.detach().cpu().numpy().argmax(axis=1))
                    true_lbls.extend(true_lbl.detach().cpu().numpy())
                    acc = metrics.accuracy_score(y_true=true_lbls, y_pred=pred_lbls)
                else:
                    loss = loss_fn(X_pred, X)

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                postfixes = {f'train_{cfg.emb_model.loss}_loss': total_loss / idx}
                if acc is not None:
                    postfixes.update({'train_acc': acc})

                t.set_postfix(**postfixes)
                t.update()

        return {'acc': acc, 'loss': total_loss / len(train_loader)}
    
    def val(current_epoch):
        total_loss = 0
        pred_lbls = []
        true_lbls = []
        with tqdm(total=len(val_loader), desc=f'{current_epoch}/{epochs}') as t:
            for idx, batch in enumerate(val_loader, start=1):
                _, X, targets = batch
                X = X.cuda()
                X_pred = model(X)

                acc = None
                if supervised:
                    targets = targets.cuda()
                    loss = loss_fn(X_pred, targets)
                    pred_lbls.extend(X_pred.detach().cpu().numpy().argmax(axis=1))
                    true_lbls.extend(targets.detach().cpu().numpy())
                    acc = metrics.accuracy_score(y_true=true_lbls, y_pred=pred_lbls)
                else:
                    loss = loss_fn(X_pred, X)

                total_loss += loss.item()
                
                postfixes = {f'val_{cfg.emb_model.loss}_loss': total_loss / idx}

                if acc is not None:
                    postfixes.update({'val_acc': acc})

                t.set_postfix(**postfixes)
                t.update()
                
        return {'acc': acc, 'loss': total_loss / len(val_loader)}
    
    for epoch in range(1, epochs + 1):
        wandb_dict = {}
        train_data = train_one_step(epoch)
        wandb_dict[f'emb_model/train_{cfg.emb_model.loss}_loss'] = train_data['loss']
        if supervised:
            wandb_dict[f'emb_model/train_acc'] = train_data['acc']

        if val_loader is not None:
            val_data = val(epoch)
            wandb_dict[f'emb_model/val_{cfg.emb_model.loss}_loss'] = val_data['loss']
            if supervised:
                wandb_dict[f'emb_model/val_acc'] = val_data['acc']
        
        wandb.log(wandb_dict)
    
    return model
        

def get_embeddings(model, dataloader, index=0, key='feats'):
    embs = []
    glb_idxes = []
    returning_targets = []
    with tqdm(total=len(dataloader), desc='Getting embeddings...') as t:
        for idx, batch in enumerate(dataloader):
            img_idx, X, targets = batch
            if type(X) == list:
                X = X[index]
            X = X.cuda()
            batch_emb = model(X)
            if type(batch_emb) == dict:
                batch_emb = batch_emb[key]
            elif type(batch_emb) == tuple:
                complete_batch_emb, batch_emb = batch_emb
            embs.append(batch_emb.detach().cpu().numpy())
            glb_idxes.append(img_idx.numpy())
            returning_targets.append(targets.numpy())
            t.update()
    embs = np.concatenate(embs)
    glb_idxes = np.concatenate(glb_idxes)
    returning_targets = np.concatenate(returning_targets)
    return {'embs': embs, 'targets': returning_targets, 'glb_idxes': glb_idxes}

def get_sim_matrix(embeddings, k=2048, gpu=True):
    d = embeddings.shape[-1]
    if gpu:
        try:
            cpu_index = faiss.IndexFlatL2(d)
            final_index = faiss.index_cpu_to_all_gpus(cpu_index)

            final_index.add(embeddings)
            print('Using GPU for NN!! Thanks FAISS! :)')
            print(final_index.ntotal)
        except:
            cpu_index = faiss.IndexFlatL2(d)
            print('No gpus for faiss! :( ')
            final_index = cpu_index
            final_index.add(embeddings)
    else:
        cpu_index = faiss.IndexFlatL2(d)
        print('No gpus for faiss! :( ')
        final_index = cpu_index
        final_index.add(embeddings)

    D, I = final_index.search(embeddings, k) # actual search
    
    return D, I

def get_clusters(embeddings, k=100, gpu=True):
    n, d = embeddings.shape
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    if gpu:
        try:
            kmeans = faiss.Kmeans(d=d, k=k, gpu=gpu)
            kmeans.train(embeddings)
            print('Thanks faiss for the gpu usage! :)')
        except:
            print("Couldn't use gpu for faiss! :(")
            kmeans = faiss.Kmeans(d=d, k=k, gpu=False)
            kmeans.train(embeddings)
    else:
        print('No gpu for faiss! :(')
        kmeans = faiss.Kmeans(d=d, k=k, gpu=False)
        kmeans.train(embeddings)
    
    cluster_dist, cluster_labels = kmeans.index.search(embeddings, 1)
    return {'dist': cluster_dist, 
            'lbls': cluster_labels}
    
def get_louvain_clusters_weighted(nn_matrix, dist_matrix, seed=None, threshold=10000):
    import networkx as nx
    import networkx.algorithms.community as nx_comm
    print(f'Using threshold for weighted Louvain clustering = {threshold}')
    no_nodes = nn_matrix.shape[0]
    knn_graph = nx.Graph()
    knn_graph.add_nodes_from(range(0, no_nodes))
    degree_one_nodes = 0
    with tqdm(total=len(nn_matrix), desc='Creating knn graph...') as t:
        for i, row in enumerate(nn_matrix):
            if len(dist_matrix[i][dist_matrix[i] <= threshold]) < 2:
                row = row[:2]
                dist_row = dist_matrix[i][:2]
                degree_one_nodes += 1
            else:
                row = row[dist_matrix[i] <= threshold]
                dist_row = dist_matrix[i][dist_matrix[i] <= threshold]
            
            for j, nn in enumerate(row):
                knn_graph.add_edge(i, nn, weight= -1 * dist_row[j]) # weights represent strength of connection
            
            t.update()

    print(f'Using KNN_graph with {(degree_one_nodes / no_nodes) * 100} pecent degree-one nodes')
    communities = nx_comm.louvain_communities(knn_graph, seed=seed)
    labels = {i: -1 for i in range(no_nodes)}
    for comm_id, comm in enumerate(communities):
        for i in comm:
            assert labels[i] == -1
            labels[i] = comm_id
    
    cluster_lbls = np.array(list(labels.values()))

    return cluster_lbls, knn_graph


def get_louvain_clusters_unweighted(nn_matrix, dist_matrix, seed=None, k=6):
    import networkx as nx
    import networkx.algorithms.community as nx_comm
    no_nodes = nn_matrix.shape[0]
    knn_graph = nx.Graph()
    knn_graph.add_nodes_from(range(0, no_nodes))
    with tqdm(total=len(nn_matrix), desc='Creating knn graph...') as t:
        for i, row in enumerate(nn_matrix):
            for j, nn in enumerate(row[:k]):
                knn_graph.add_edge(i, nn)
                
            t.update()

    communities = nx_comm.louvain_communities(knn_graph, seed=seed)
    labels = {i: -1 for i in range(no_nodes)}
    for comm_id, comm in enumerate(communities):
        for i in comm:
            assert labels[i] == -1
            labels[i] = comm_id
    
    cluster_lbls = np.array(list(labels.values()))

    return cluster_lbls, knn_graph



def load_npy(path):
    return np.load(path)

def save_npy(data, path):
    np.save(path, data)
    return True

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pil_image(np_array):
    from einops import rearrange
    if len(np_array.shape) == 3 and np_array.shape[-1] != 3:
        np_array = rearrange(np_array, 'c h w -> h w c')
    return Image.fromarray(np.uint8(np_array)).convert('RGB')

def create_nns(best_nns, best_nn_ids, save_path, dataset_data):
    """
        dataset_data: numpy array of imgs in numpy format
    """
    for i in range(len(best_nns)):
        make_dirs(os.path.join(save_path, f'{best_nn_ids[i]}'))
        for idx, j in enumerate(best_nns[i]):
            if type(dataset_data[j]) == np.ndarray:
                img = pil_image(dataset_data[j])
            else:
                img = Image.open(dataset_data[j]).convert("RGB")
            img.save(os.path.join(save_path, f'{best_nn_ids[i]}', f'{idx}_{j}.png'))


def check_nns(embeddings, dataset, save_path, k=5, random_ids=None):
    emb_dist_matrix, emb_sim_matrix = get_sim_matrix(embeddings)
    best_k = emb_sim_matrix[:, :k + 1]
    if random_ids is None:
        ids = [i for i in range(len(best_k))]
        random_ids = np.random.choice(ids, 50)

    best_k_random = best_k[random_ids]
    create_nns(best_nns=best_k_random ,best_nn_ids=random_ids, save_path=save_path, dataset_data=dataset)
    return random_ids

def subsample_dataset(dataset, subsample_by):
    from torchvision import datasets
    # currently only for inat

    dataset_type = type(dataset).__bases__[0]
    print(dataset_type)
    if dataset_type is not datasets.INaturalist and \
        type(dataset) is not datasets.INaturalist:
        print(f'Subsampling not supported for {dataset_type} or {type(dataset)}')
        return dataset
    
    labels = np.array(list(list(zip(*dataset.index))[0]))
    imgs = np.array(list(list(zip(*dataset.index))[1]))
    no_classes = len(np.unique(labels))
    assert no_classes > labels.max()
    new_no_classes = no_classes // subsample_by
    new_imgs = imgs[labels <= new_no_classes]
    new_labels = labels[labels <= new_no_classes]
    new_index = list(zip(new_labels, new_imgs))
    dataset.index = new_index
    return dataset

def _dict_add_value_dict(config_dict):
    d = dict()
    for k, v in config_dict.items():
        d[k] = dict(desc=None, value=v)
    return d

def _config_telemetry_update(experiment, config_dict: Dict[str, Any]) -> None:
        from wandb.sdk.lib import proto_util, config_util
        """Add legacy telemetry to config object."""
        wandb_key = "_wandb"
        config_dict.setdefault(wandb_key, dict())
        s: str
        b: bool
        s = experiment._telemetry_obj.python_version
        if s:
            config_dict[wandb_key]["python_version"] = s
        s = experiment._telemetry_obj.cli_version
        if s:
            config_dict[wandb_key]["cli_version"] = s
        # s = self._telemetry_get_framework()
        # if s:
        #     config_dict[wandb_key]["framework"] = s
        s = experiment._telemetry_obj.huggingface_version
        if s:
            config_dict[wandb_key]["huggingface_version"] = s
        b = experiment._telemetry_obj.env.jupyter
        config_dict[wandb_key]["is_jupyter_run"] = b
        b = experiment._telemetry_obj.env.kaggle
        config_dict[wandb_key]["is_kaggle_kernel"] = b

        config_dict[wandb_key]["start_time"] = experiment._start_time
        
        t = proto_util.proto_encode_to_dict(experiment._telemetry_obj)
        config_dict[wandb_key]["t"] = t
    
# def _config_metric_update(self, config_dict: Dict[str, Any]) -> None:
#     """Add default xaxis to config."""
#     if not self._config_metric_pbdict_list:
#         return
#     wandb_key = "_wandb"
#     config_dict.setdefault(wandb_key, dict())
#     config_dict[wandb_key]["m"] = self._config_metric_pbdict_list

def _config_save(experiment, config_value_dict) -> None:
        from wandb.sdk.lib import proto_util, config_util
        config_path = os.path.join(experiment._settings.files_dir, "config.yaml")
        config_util.save_config_file_from_dict(config_path, config_value_dict)

def handle_wandb_offline(wandb_logger):
    config_file = wandb_logger._experiment.config.as_dict()
    _config_telemetry_update(wandb_logger._experiment, config_file)
    # self._config_metric_update(config_dict)
    config_file = _dict_add_value_dict(config_file)
    _config_save(wandb_logger._experiment, config_file)

def create_pos_neg_hist_plot(dataset_name, emb_sim_matrix, emb_dist_matrix, lbls, k, bins=300, save_path='./', true_lbls=None):
    plt.clf()
    # sim_matrix = emb_sim_matrix[:, 1:k]
    sim_matrix = emb_sim_matrix[:, :k]
    all_lbls_sim_matrix = lbls[sim_matrix]
    # all_lbls_true = lbls.repeat(k - 1).reshape(-1, k - 1)
    if true_lbls is None:
        all_lbls_true = lbls.repeat(k).reshape(-1, k)
    else:
        all_lbls_true = true_lbls.repeat(k).reshape(-1, k)
    correct_lbls = (all_lbls_true == all_lbls_sim_matrix)
    # dist_matrix = emb_dist_matrix[:, 1:k]
    dist_matrix = emb_dist_matrix[:, :k]
    plt.hist(dist_matrix[np.logical_not(correct_lbls)], bins=bins, color='r', alpha=0.3)
    plt.hist(dist_matrix[correct_lbls], bins=bins, color='g', alpha=0.3)
    plt.title(f'{dataset_name} k = {k}')
    plt.savefig(os.path.join(save_path, f'{dataset_name}_k{k}.pdf'))
    plt.clf()
    plt.hist(dist_matrix[np.logical_not(correct_lbls)], bins=bins, color='r', alpha=0.3)
    plt.hist(dist_matrix[correct_lbls], bins=bins, color='g', alpha=0.3)
    plt.yscale('log')
    plt.title(f'{dataset_name} k = {k} Log Scale')
    plt.savefig(os.path.join(save_path, f'{dataset_name}_k{k}_log.pdf'))

def create_pos_neg_hist_plot_from_neg_and_pos(dataset_name, pos_dists, neg_dists, bins=300):
    plt.clf()
    plt.hist(neg_dists, bins=bins, color='r', alpha=0.3)
    plt.hist(pos_dists, bins=bins, color='g', alpha=0.3)
    plt.title(f'{dataset_name} all')
    plt.savefig(f'{dataset_name}_all.pdf')
    plt.clf()
    plt.hist(neg_dists, bins=bins, color='r', alpha=0.3)
    plt.hist(pos_dists, bins=bins, color='g', alpha=0.3)
    plt.yscale('log')
    plt.title(f'{dataset_name} all Log Scale')
    plt.savefig(f'{dataset_name}_all_log.pdf')

def plot_sim_histogram(dataset_name, sims, labels, bins=300, save_path='./', pos_mask=None):
    if pos_mask is None:
        labels1 = labels.repeat(sims.shape[0]).reshape(-1, sims.shape[0])
        labels_t = torch.tensor(labels)
        labels2 = labels_t.repeat(sims.shape[0]).reshape(sims.shape[0], -1).numpy()
        pos_mask = (labels1 == labels2)

    plt.clf()
    print('Started plotting first')
    plt.hist(sims[np.logical_not(pos_mask)], bins=bins, color='r', alpha=0.3)
    plt.hist(sims[pos_mask], bins=bins, color='g', alpha=0.3)
    plt.title(f'{dataset_name}')
    plt.savefig(os.path.join(save_path, f'{dataset_name}_all.pdf'))
    plt.clf()
    print('Started plotting second')
    plt.hist(sims[np.logical_not(pos_mask)], bins=bins, color='r', alpha=0.3)
    plt.hist(sims[pos_mask], bins=bins, color='g', alpha=0.3)
    plt.yscale('log')
    plt.title(f'{dataset_name} Log Scale')
    plt.savefig(os.path.join(save_path, f'{dataset_name}_all_log.pdf'))
    return pos_mask

# def dict_from_proto_list(obj_list):
#     d = dict()
#     for item in obj_list:
#         d[item.key] = dict(desc=None, value=json.loads(item.value_json))
#     return d

class PlotEmbeddingsCallback(Callback):
    """
        Plot global pairwise cosine similarities for both GPS-SSL and NNCLR
    """
    def __init__(self, dataset_name, data_loader, save_path, **kwargs):
        super().__init__()
        self.epoch = 0
        self.dataset_name = dataset_name
        self.data_loader = data_loader
        self.save_path = save_path
        self.pos_mask = None
        self.kwargs = kwargs

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch += 1
        output = get_embeddings(pl_module, self.data_loader, **self.kwargs)
        embeddings = output['embs']
        embedding_labels = output['targets']

        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sims = np.matmul(embeddings_norm, embeddings_norm.T)
        pos_mask = plot_sim_histogram(self.dataset_name + f'_ep{self.epoch}', sims, embedding_labels, bins=50, save_path=self.save_path, pos_mask=self.pos_mask)
        if self.pos_mask is None:
            self.pos_mask = pos_mask
        

class ClassNNPecentageCallback(Callback):
    def __init__(self, dataset_name,
                 data_loader, 
                 save_path, 
                 plot_nearest_neighbors=False,
                 key='feats'):
        super().__init__()
        self.epoch = 0
        self.data_loader = data_loader
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.plot_nearest_neighbors = plot_nearest_neighbors
        self.key = key

    def on_train_epoch_start(self, trainer, pl_module):
        for logger in trainer.loggers:
            percentage_metrics = trainer.train_dataloader.loaders.dataset.relevant_classes
            not_from_cluster_percentage_metrics = trainer.train_dataloader.loaders.dataset.not_from_cluster_percentage
            no_nns_metrics = trainer.train_dataloader.loaders.dataset.no_nns
            nn_threshold = trainer.train_dataloader.loaders.dataset.nn_threshold
            extra_info = trainer.train_dataloader.loaders.dataset.extra_info
            metrics_to_log = {'relevant_class_percentage_AVG': percentage_metrics['avg'],
                                'relevant_class_percentage_MEDIAN': percentage_metrics['median'],
                                'relevant_class_percentage_VAR': percentage_metrics['var'],
                                'not_from_cluster_percentage_AVG': not_from_cluster_percentage_metrics['avg'],
                                'not_from_cluster_percentage_MEDIAN': not_from_cluster_percentage_metrics['median'],
                                'not_from_cluster_percentage_VAR': not_from_cluster_percentage_metrics['var'],
                                'no_nns_metrics_AVG': no_nns_metrics['avg'],
                                'no_nns_metrics_MEDIAN': no_nns_metrics['median'],
                                'no_nns_metrics_VAR': no_nns_metrics['var'],
                                'no_nns_metrics_MAX': no_nns_metrics['max'],
                                'no_nns_metrics_MIN': no_nns_metrics['min'],
                                'nn_threshold': nn_threshold}
            
            metrics_to_log.update(extra_info)

            logger.log_metrics(metrics_to_log, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

        self.epoch += 1
        if self.plot_nearest_neighbors:
            output = get_embeddings(pl_module, self.data_loader, key=self.key)
            embeddings = output['embs']
            embedding_labels = output['targets']

            d = embeddings.shape[1]
            # if torch.cuda.is_available():
            #     try:
            #         cpu_index = faiss.IndexFlatL2(d)
            #         final_index = faiss.index_cpu_to_all_gpus(cpu_index)
            #         final_index.add(embeddings)
            #     except:
            #         cpu_index = faiss.IndexFlatL2(d)
            #         print('No gpus for faiss! :( ')
            #         final_index = cpu_index
            #         final_index.add(embeddings)
            # else:
            cpu_index = faiss.IndexFlatL2(d)
            print('No gpus avaialble for faiss! :((( ')
            final_index = cpu_index
            final_index.add(embeddings)

            D, I = final_index.search(embeddings, k=200) # actual search

            for k in [5, 25, 100]:
                print(f'creating plot for k = {k}...')
                create_pos_neg_hist_plot(f"{self.dataset_name}_ep{self.epoch}",
                                        emb_sim_matrix=I,
                                        emb_dist_matrix=D,
                                        lbls=embedding_labels, 
                                        k=k, 
                                        bins=300, 
                                        save_path=self.save_path,
                                        true_lbls=embedding_labels)

class ClassNNPecentageCallback_NNCLR(Callback):
    def __init__(self, dataset_name, save_path):
        super().__init__()
        self.epoch = 0
        self.dataset_name = dataset_name
        self.save_path = save_path

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch += 1
        output = get_embeddings(pl_module, trainer.train_dataloader, index=0, key='z')
        embeddings = output['embs']
        embedding_labels = output['targets']
        glb_idxes = output['glb_idxes']
        queue = pl_module.queue.cpu().numpy()
        queue_y = pl_module.queue_y.cpu().numpy()
        queue_idx = pl_module.queue_idx.cpu().numpy()

        # dists = torch.matmul(pl_module.queue, pl_module.queue.T).cpu().numpy()

        d = queue.shape[1]
        if torch.cuda.is_available():
            try:
                cpu_index = faiss.IndexFlatL2(d)
                final_index = faiss.index_cpu_to_all_gpus(cpu_index)
                final_index.add(queue)
            except:
                cpu_index = faiss.IndexFlatL2(d)
                print('No gpus for faiss! :( ')
                final_index = cpu_index
                final_index.add(queue)
        else:
            cpu_index = faiss.IndexFlatL2(d)
            print('No gpus avaialble for faiss! :((( ')
            final_index = cpu_index
            final_index.add(queue)

        D, I = final_index.search(embeddings, k=200) # actual search
        
        nearest_neighbor_queue_idxes = queue_idx[I[:, 0]].flatten()
        embedding_idxes = glb_idxes.flatten()

        nearest_neighbor_queue_labels = queue_y[I]
        embedding_labels_reshaped = np.repeat(embedding_labels, I.shape[1]).reshape(-1, I.shape[1])
        
        same_img_percentage = (embedding_idxes == nearest_neighbor_queue_idxes).mean()
        
        relevant_class_percentage_AVG = (embedding_labels_reshaped == nearest_neighbor_queue_labels).mean()
        relevant_class_percentage_VAR = np.var((embedding_labels_reshaped == nearest_neighbor_queue_labels).sum(axis=1))
        relevant_class_percentage_MEDIAN = np.median((embedding_labels_reshaped == nearest_neighbor_queue_labels).sum(axis=1))

        for k in [5, 25, 100]:
            print(f'creating plot for k = {k}...')
            create_pos_neg_hist_plot(f"{self.dataset_name}_ep{self.epoch}",
                                     emb_sim_matrix=I,
                                     emb_dist_matrix=D,
                                     lbls=queue_y, 
                                     k=k, 
                                     bins=300, 
                                     save_path=self.save_path,
                                     true_lbls=embedding_labels)


        for logger in trainer.loggers:
        #     percentage_metrics = trainer.train_dataloader.loaders.dataset.relevant_classes
        #     not_from_cluster_percentage_metrics = trainer.train_dataloader.loaders.dataset.not_from_cluster_percentage
        #     no_nns_metrics = trainer.train_dataloader.loaders.dataset.no_nns
        #     nn_threshold = trainer.train_dataloader.loaders.dataset.nn_threshold
        #     extra_info = trainer.train_dataloader.loaders.dataset.extra_info
            metrics_to_log = {'relevant_class_percentage_AVG': relevant_class_percentage_AVG,
                                'relevant_class_percentage_MEDIAN': relevant_class_percentage_MEDIAN,
                                'relevant_class_percentage_VAR': relevant_class_percentage_VAR,
                                'same_img_percentage': same_img_percentage}
        #                         'not_from_cluster_percentage_AVG': not_from_cluster_percentage_metrics['avg'],
        #                         'not_from_cluster_percentage_MEDIAN': not_from_cluster_percentage_metrics['median'],
        #                         'not_from_cluster_percentage_VAR': not_from_cluster_percentage_metrics['var'],
        #                         'no_nns_metrics_AVG': no_nns_metrics['avg'],
        #                         'no_nns_metrics_MEDIAN': no_nns_metrics['median'],
        #                         'no_nns_metrics_VAR': no_nns_metrics['var'],
        #                         'no_nns_metrics_MAX': no_nns_metrics['max'],
        #                         'no_nns_metrics_MIN': no_nns_metrics['min'],
        #                         'nn_threshold': nn_threshold}
            
            logger.log_metrics(metrics_to_log, step=trainer.fit_loop.epoch_loop._batches_that_stepped)


def get_clip_embeddings(model, dataloader, device):
    embeddings = []
    dl_pb = tqdm(dataloader)
    for idx, batch in enumerate(dl_pb):
        x, y = batch
        x = x.to(device)
        out = model.encode_image(x)
        embeddings.append(out.detach().cpu().numpy())    
    return np.concatenate(embeddings)
