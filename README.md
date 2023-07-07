# Guided Nearest Neighbor Contrastive Learning with Prior Knowledge For Hotel Recognition
Repository for the "Guided Nearest Neighbor Contrastive Learning with Prior Knowledge For Hotel Recognition" Paper, submitted to ICDM 2023.

RHID (Revised HotelID-2021) can be downloaded from [here](google.com).

Example of running the code:
```
python main_nn.py --config-path=scripts/pretrain_pk/rhid-test/ --config-name=gnnclr.yaml augmentations=reconstruction backbone.name=resnet18 data.num_nns=1 data.num_nns_choice=5 data.reload_freq=0 emb_model=resnet50 max_epochs=200 nn_key=feats optimizer.classifier_lr=0.3 optimizer.lr=0.1 optimizer.weight_decay=0.001
```