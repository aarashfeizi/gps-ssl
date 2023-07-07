import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, cfg) -> None:
        """
            cfg.ae_input_size: size of the images (int)
            cfg.ae_sizes: size of the layers for the encoder and decoder (list of ints)
        """
        super().__init__()
        encoder_layers = []
        dencoder_layers = []
        layers_sizes = [cfg.emb_model.input_size]
        layers_sizes.extend(cfg.emb_model.sizes)
        self.is_train = False
        for idx in range(1, len(layers_sizes)):
            layer = nn.Linear(layers_sizes[idx - 1], layers_sizes[idx])
            relu = nn.ReLU()
            encoder_layers.extend([layer, relu])

        encoder_layers = encoder_layers[:-1] # remove last relu from encoder

        for idx in range(len(layers_sizes) - 1, 0, -1):
            layer = nn.Linear(layers_sizes[idx], layers_sizes[idx - 1])
            relu = nn.ReLU()
            dencoder_layers.extend([layer, relu])
        
        dencoder_layers = dencoder_layers[:-1] # remove last relu from dencoder

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*dencoder_layers)
    
    def train(self):
        super()
        self.is_train = True
    
    def eval(self):
        super()
        self.is_train = False

    def forward(self, x):
        x_shape = x.shape
        latent = self.encoder(x.reshape(x_shape[0], -1)) #batch and rest
        if self.is_train:
            x_pred = self.decoder(latent)
            x_pred = x_pred.reshape(*x.shape)
            return x_pred
        else:
            return latent