from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import  ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
import torch.nn as nn
import torch
import os

RESNETS = {
    'resnet18': resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    'resnet50': resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
    'resnet101': resnet101(weights=ResNet101_Weights.IMAGENET1K_V2),
}

RESNETS_RANDOM = {
    'resnet18': resnet18(weights=None),
    'resnet50': resnet50(weights=None),
    'resnet101': resnet101(weights=None),
}

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        self.outputs = cfg.emb_model.outputs # default: 4 
        self.get_extended_features = cfg.emb_model.get_extended_features

        if cfg.emb_model.pretrained == 'false':
            backbone = RESNETS_RANDOM[cfg.emb_model.name]
        elif cfg.emb_model.pretrained == 'true' or cfg.emb_model.pretrained == 'imagenet':
            backbone = RESNETS[cfg.emb_model.name]
        else:
            backbone = RESNETS_RANDOM[cfg.emb_model.name]
            model_path = os.path.join(cfg.emb_model.ckpt_path, f'{cfg.emb_model.name}_{cfg.emb_model.pretrained}.ckpt')

            print(f'Loading {model_path}')
            assert os.path.exists(model_path), f'{model_path} does not exist! :('
            checkpoint = torch.load(model_path)

            new_ckpt_dict = self.__fix_keys(checkpoint['state_dict'], 'backbone.', '')
            
            mk = backbone.load_state_dict(new_ckpt_dict, strict=False)
            assert set(mk.missing_keys) == {'fc.weight', 'fc.bias'}, f'Missing keys are {mk.missing_keys}'
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc
        
        if cfg.emb_model.train:
            if cfg.emb_model.train_method == 'supervised':
                in_features = self.fc.in_features
                self.fc = nn.Linear(in_features=in_features, out_features=cfg.data.num_classes)
            else:
                raise Exception('Unsupported training method')
            
            self.train()
        else:
            self.fc = nn.Identity()
            self.eval()

    def _add_to_out(self, out, new_output):
        if out is None:
            return [torch.flatten(new_output.detach().cpu(), 1)]
        else:
            out.append(torch.flatten(new_output.detach().cpu(), 1))
            return out
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat_out = None
        x = self.layer1(x)
        if 1 in self.outputs:
            feat_out = self._add_to_out(feat_out, self.avgpool(x))
        x = self.layer2(x)
        if 2 in self.outputs:
            feat_out = self._add_to_out(feat_out, self.avgpool(x))
        x = self.layer3(x)
        if 3 in self.outputs:
            feat_out = self._add_to_out(feat_out, self.avgpool(x))
        x = self.layer4(x)
        if 4 in self.outputs:
            feat_out = self._add_to_out(feat_out, self.avgpool(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        feat_out = torch.cat(feat_out, 1)

        if self.get_extended_features:
            return out, feat_out
        else:
            return out
    
    def eval(self):
        self.train(False)
        self.fc = nn.Identity()
    
    def __fix_keys(self, d, old, new):
        new_d = {}
        for k, v in d.items():
            new_k = k.replace(old, new)
            new_d[new_k] = v
        
        new_d.pop('fc.weight')
        new_d.pop('fc.bias')
        return new_d