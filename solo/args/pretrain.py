import os

import omegaconf
from omegaconf import OmegaConf
from solo.utils.auto_resumer import AutoResumer
from solo.utils.auto_umap import AutoUMAP
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import omegaconf_select
import argparse

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule
except ImportError:
    _dali_available = False
else:
    _dali_available = True

_N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "svhn": 10,
    "aircrafts": 100,
    "cub": 100,
    "inat": 10000,
    "pets": 37,
    "dtd": 47,
    "imagenet": 1000,
    "imagenet100": 100,
    "rhid-val": 3150,
    "rhid-test": 4406,
    "hotels50k-test": 18918
}


_SUPPORTED_DATASETS = [
    "cifar10",
    "cifar100",
    "stl10",
    "svhn",
    "aircrafts",
    "cub",
    "inat",
    "pets",
    "dtd",
    "imagenet",
    "imagenet100",
    "rhid-val",
    "rhid-test", 
    "hotels50k-test",
    "custom",
]


def add_and_assert_dataset_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for dataset config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    assert not OmegaConf.is_missing(cfg, "data.dataset")
    assert not OmegaConf.is_missing(cfg, "data.train_path")

    assert cfg.data.dataset in _SUPPORTED_DATASETS

    # if validation path is not available, assume that we want to skip eval
    cfg.data.val_path = omegaconf_select(cfg, "data.val_path", None)
    cfg.data.format = omegaconf_select(cfg, "data.format", "image_folder")
    cfg.data.no_labels = omegaconf_select(cfg, "data.no_labels", False)
    cfg.data.fraction = omegaconf_select(cfg, "data.fraction", -1)
    cfg.data.reload_freq = omegaconf_select(cfg, "data.reload_freq", 0)
    cfg.data.emb_path = omegaconf_select(cfg, "data.emb_path", None)
    cfg.debug_augmentations = omegaconf_select(cfg, "debug_augmentations", False)

    return cfg


def add_and_assert_wandb_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for wandb config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.wandb = omegaconf_select(cfg, "wandb", {})
    cfg.wandb.enabled = omegaconf_select(cfg, "wandb.enabled", False)
    cfg.wandb.entity = omegaconf_select(cfg, "wandb.entity", None)
    cfg.wandb.project = omegaconf_select(cfg, "wandb.project", "solo-learn")
    cfg.wandb.offline = omegaconf_select(cfg, "wandb.offline", os.environ.get('WANDB_MODE') == 'offline')
    cfg.wandb.save_dir = omegaconf_select(cfg, "wandb.save_dir", 'wandb/')
    cfg.wandb.tags = omegaconf_select(cfg, "wandb.tags", None)
    

    return cfg


def add_and_assert_lightning_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for Pytorch Lightning config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.seed = omegaconf_select(cfg, "seed", 5)
    cfg.resume_from_checkpoint = omegaconf_select(cfg, "resume_from_checkpoint", None)
    cfg.strategy = omegaconf_select(cfg, "strategy", None)

    return cfg

def add_and_assert_scheduler(cfg):
    cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
    cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
    cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
    cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
    cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "step")


    return cfg


def add_and_assert_gps_cfg(cfg):
    cfg.gps = omegaconf_select(cfg, "gps", False)
    cfg.nn_key = omegaconf_select(cfg, "nn_key", 'feats')
    cfg.log_path = omegaconf_select(cfg, "log_path", '../../scratch/ht-image-ssl/logs/')
    cfg.data.num_nns = omegaconf_select(cfg, "data.num_nns", 1)
    cfg.data.num_nns_choice = omegaconf_select(cfg, "data.num_nns_choice", 1)
    cfg.data.subsample_by = omegaconf_select(cfg, "data.subsample_by", 1)
    cfg.data.filter_sim_matrix = omegaconf_select(cfg, "data.filter_sim_matrix", False)
    cfg.data.num_clusters = omegaconf_select(cfg, "data.num_clusters", 1)
    cfg.data.clustering_algo = omegaconf_select(cfg, "data.clustering_algo", 'kmeans') # ['kmeans', 'louvainW', 'louvainU']
    cfg.data.cluster_louvain = omegaconf_select(cfg, "data.cluster_louvain", False)
    cfg.data.nn_threshold = omegaconf_select(cfg, "data.nn_threshold", -1)
    cfg.data.threshold_mode = omegaconf_select(cfg, "data.threshold_mode", "fixed") # ['fixed', 'adaptive']
    cfg.data.threshold_k = omegaconf_select(cfg, "data.threshold_k", 20) # e.g., 20, 5, etc.
    cfg.data.threshold_mode_type = omegaconf_select(cfg, "data.threshold_mode_type", "mean+std") # ['mean+std', 'mean', 'mean-std']
    cfg.data.plot_distances = omegaconf_select(cfg, "data.plot_distances", False) # [True, False]
    cfg.data.plot_distances_after_epoch = omegaconf_select(cfg, "data.plot_distances_after_epoch", False) # [True, False]
    cfg.data.nn_hop = omegaconf_select(cfg, "data.nn_hop", 0) # {0, 1, 2, ...}
    
    cfg.scheduler_max_epochs = omegaconf_select(cfg, "scheduler_max_epochs", -1)
    
    
    
    assert cfg.data.num_nns_choice >= cfg.data.num_nns
    
    print('Log path is: ', cfg.log_path)
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)

    if cfg.gps:
        cfg.emb_model = omegaconf_select(cfg, "emb_model", {})
        cfg.emb_model.name = omegaconf_select(cfg, "emb_model.name", "resnet50")
        cfg.emb_model.train = omegaconf_select(cfg, "emb_model.train", False)
        cfg.emb_model.epochs = omegaconf_select(cfg, "emb_model.epochs", 0)
        cfg.emb_model.loss = omegaconf_select(cfg, "emb_model.loss", 'mse')
        cfg.emb_model.opt = omegaconf_select(cfg, "emb_model.opt", "adam")
        cfg.emb_model.sizes = omegaconf_select(cfg, "emb_model.sizes", [])
        cfg.emb_model.kernel_sizes = omegaconf_select(cfg, "emb_model.kernel_sizes", [3 for _ in cfg.emb_model.sizes])
        cfg.emb_model.input_size = omegaconf_select(cfg, "emb_model.input_size", 32)
        cfg.emb_model.lr = omegaconf_select(cfg, "emb_model.lr", 1e-3)
        cfg.emb_model.weight_decay = omegaconf_select(cfg, "emb_model.weight_decay", 1e-5)
        cfg.emb_model.random_ids = omegaconf_select(cfg, "emb_model.random_ids", None)
        cfg.emb_model.pretrained = omegaconf_select(cfg, "emb_model.pretrained", 'true')
        cfg.emb_model.train_method = omegaconf_select(cfg, "emb_model.train_method", 'supervised')
        cfg.emb_model.transform = omegaconf_select(cfg, "emb_model.transform", 'noTransform')
        cfg.emb_model.supervised = omegaconf_select(cfg, "emb_model.supervised", False)
        cfg.emb_model.ckpt_path = omegaconf_select(cfg, "emb_model.ckpt_path", './emb_model_checkpoints')
        cfg.emb_model.outputs = omegaconf_select(cfg, "emb_model.outputs", [4]) # default: 4 
        cfg.emb_model.get_extended_features = omegaconf_select(cfg, "emb_model.get_extended_features", False)

        cfg.emb_model.outputs = [num for num in cfg.emb_model.outputs]

    
        

    
    return cfg

def parse_cfg(cfg: omegaconf.DictConfig):
    # default values for checkpointer
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)

    # default values for auto_resume
    cfg = AutoResumer.add_and_assert_specific_cfg(cfg)

    # default values for auto_umap
    cfg = AutoUMAP.add_and_assert_specific_cfg(cfg)

    # default values for dali
    if _dali_available:
        cfg = PretrainDALIDataModule.add_and_assert_specific_cfg(cfg)

    # assert dataset parameters
    cfg = add_and_assert_dataset_cfg(cfg)
    
    # augmentations for calculating the nearest neighbors
    cfg.nn_augmentations = omegaconf_select(cfg, "nn_augmentations", 'no_transform')
    
    cfg = add_and_assert_gps_cfg(cfg)

    cfg = add_and_assert_scheduler(cfg)

    # default values for wandb
    cfg = add_and_assert_wandb_cfg(cfg)

    # default values for pytorch lightning stuff
    cfg = add_and_assert_lightning_cfg(cfg)

    # if backbone is vit, input size must be 224 
    if cfg.backbone.name.startswith('vit'):
        for aug in cfg.augmentations:
            if aug.crop_size != 224:
                print(f'setting augmentation crop size from {aug.crop_size} to 224')
                aug.crop_size = 224

    # by default, set TEST mode to False for datasets such as Aircrafts
    cfg.test = omegaconf_select(cfg, "test", False)

    # by default, set Early Stopping Factor (i.e. max_epochs / es_factor will be es tolerance)
    cfg.es_factor = omegaconf_select(cfg, "es_facotr", 10)

    # make sure method_kwargs has normalize_projector set to False as default (this is only used in SimClr and BYOL)
    cfg.method_kwargs.normalize_projector = omegaconf_select(cfg, "method_kwargs.normalize_projector", 'false')
    
    # extra processing
    if cfg.data.dataset in _N_CLASSES_PER_DATASET:
        cfg.data.num_classes = _N_CLASSES_PER_DATASET[cfg.data.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        cfg.data.num_classes = max(
            1,
            len([entry.name for entry in os.scandir(cfg.data.train_path) if entry.is_dir]),
        )

    # find number of big/small crops
    big_size = cfg.augmentations[0].crop_size
    num_large_crops = num_small_crops = 0
    for pipeline in cfg.augmentations:
        if big_size == pipeline.crop_size:
            num_large_crops += pipeline.num_crops
        else:
            num_small_crops += pipeline.num_crops
    cfg.data.num_large_crops = num_large_crops
    if cfg.gps:
        cfg.data.num_large_crops += cfg.data.num_nns
    cfg.data.num_small_crops = num_small_crops

    if cfg.data.format == "dali":
        assert cfg.data.dataset in ["imagenet100", "imagenet", "custom"]

    # make sure checkpoint dir and wandb dir exists:
    if not os.path.exists(cfg.checkpoint_config.dir):
        os.makedirs(cfg.checkpoint_config.dir)

    if not os.path.exists(cfg.wandb.save_dir):
        os.makedirs(cfg.wandb.save_dir)

    # adjust lr according to batch size
    cfg.num_nodes = omegaconf_select(cfg, "num_nodes", 1)
    # if args.batch_size is not None:
    #     scale_factor = args.batch_size * len(cfg.devices) * cfg.num_nodes / 256
    # else:
    scale_factor = cfg.optimizer.batch_size * len(cfg.devices) * cfg.num_nodes / 256
    # if args.lr is not None:
    #     cfg.optimizer.lr = args.lr * scale_factor
    # else:
    cfg.optimizer.lr = cfg.optimizer.lr * scale_factor
    if cfg.data.val_path is not None:
        assert not OmegaConf.is_missing(cfg, "optimizer.classifier_lr")
        # if args.classifier_lr is not None:
        #     cfg.optimizer.classifier_lr = args.classifier_lr * scale_factor
        # else:
        cfg.optimizer.classifier_lr = cfg.optimizer.classifier_lr * scale_factor

    # extra optimizer kwargs
    cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
    if cfg.optimizer.name == "sgd":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
    elif cfg.optimizer.name == "lars":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
        cfg.optimizer.kwargs.eta = omegaconf_select(cfg, "optimizer.kwargs.eta", 1e-3)
        cfg.optimizer.kwargs.clip_lr = omegaconf_select(cfg, "optimizer.kwargs.clip_lr", False)
        cfg.optimizer.kwargs.exclude_bias_n_norm = omegaconf_select(
            cfg,
            "optimizer.kwargs.exclude_bias_n_norm",
            False,
        )
    elif cfg.optimizer.name == "adamw":
        cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])

    return cfg


# def get_args():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--batch_size', default=None, type=int) 
#     parser.add_argument('--classifier_lr', default=None, type=float) 
#     parser.add_argument('--lr', default=None, type=float) 
#     parser.add_argument('--method', default=None, choices=['byol', 'simsiam', 'simclr'])
#     parser.add_argument('--config-name', default=None) 
#     parser.add_argument('--config-path', default=None) 


#     args = parser.parse_args()

#     return args
