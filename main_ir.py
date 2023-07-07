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

import json
import os
from pathlib import Path
from typing import Tuple
import omegaconf

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from solo.args.ir import parse_args_ir
from solo.data.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from solo.methods import METHODS
from solo.utils.ir import ImageRetrieval


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module) -> Tuple[torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """

    model.eval()
    backbone_features, proj_features, labels = [], [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model(im)
        backbone_features.append(outs["feats"].detach())
        proj_features.append(outs["z"])
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    proj_features = torch.cat(proj_features)
    labels = torch.cat(labels)
    return backbone_features, proj_features, labels


@torch.no_grad()
def run_ir(
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int,
    distance_fx: str,
    ratios
) -> Tuple[float]:
    """Runs offline knn on a train and a test dataset.

    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
        k (int): number of neighbors.
        distance_fx (str): distance function.

    Returns:
        Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
    """

    # build knn
    ir = ImageRetrieval(
        k=k,
        distance_fx=distance_fx,
        ratios=ratios,
    )

    # add features
    ir(
        test_features=test_features,
        test_targets=test_targets,
    )

    # compute
    acc1, acc5 = ir.compute()
    ur_output = None
    if ratios is not None:
        ur_output = ir.compute_ur()
    auc_score, pred_true_dict = ir.compute_auroc(k=1)
    ir.reset()

    # free up memory
    del ir

    return acc1, acc5, auc_score, ur_output


def main():
    args = parse_args_ir()

    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)
        cfg = omegaconf.OmegaConf.create(method_args)


    # build the model
    model = METHODS[method_args["method"]](cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    # prepare data
    _, T = prepare_transforms(args.dataset, is_vit=cfg.backbone.name.startswith('vit'))
    train_data_path = os.path.join(args.data_path, args.train_name)
    for val in args.vals:
        path = os.path.split(ckpt_path)[0]
        proj_path = os.path.join(path, f'{val}_projector.npy')
        bb_path = os.path.join(path, f'{val}_backbone.npy')
        targets_path = os.path.join(path, f'{val}_targets.npy')
        if not os.path.exists(proj_path) or \
            not os.path.exists(bb_path) or \
            not os.path.exists(targets_path):
            val_data_path = os.path.join(args.data_path, val)
            train_dataset, val_dataset = prepare_datasets(
                args.dataset,
                T_train=T,
                T_val=T,
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                data_format=args.data_format,
                data_fraction=args.data_fraction,
                test=args.test,
                is_classification=False
            )

            _, val_loader = prepare_dataloaders(
                train_dataset,
                val_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            # # extract train features
            # train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
            # train_features = {"backbone": train_features_bb, "projector": train_features_proj}

            # extract test features
            test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
            
            print(f'Caching to {path}')
            np.save(bb_path, test_features_bb.cpu().numpy())
            np.save(proj_path, test_features_proj.cpu().numpy())
            np.save(targets_path, test_targets.cpu().numpy())
        else:
            print(f'Loading from {path}')
            test_features_bb = torch.tensor(np.load(bb_path))
            test_features_proj = torch.tensor(np.load(proj_path))
            test_targets = torch.tensor(np.load(targets_path))
            
        test_features = {"backbone": test_features_bb, "projector": test_features_proj}

        # run k-nn for all possible combinations of parameters
        for feat_type in args.feature_type:
            print(f"\n### {feat_type.upper()} ###")
            for distance_fx in args.distance_function:
                print("---")
                print(f"Running Image Retrieval with params: distance_fx={distance_fx}...")
                acc1, acc5, auroc, ur_output = run_ir(
                    test_features=test_features[feat_type],
                    test_targets=test_targets,
                    k=20,
                    distance_fx=distance_fx,
                    ratios=args.ratios
                )
                print(f"Result on {val}: acc@1 = {acc1}, acc@5 = {acc5}, auroc = {auroc}, under_rep_output = {ur_output}")


if __name__ == "__main__":
    main()
