from solo.emb_methods.resnet import ResNet
from solo.emb_methods.autoencoder import AE
from solo.emb_methods.conv_autoencoder import CAE

EMB_METHODS = {
    "resnet": ResNet,
    "autoencoder": AE, 
    "conv_autoencoder": CAE,

}
__all__ = [
    "ResNet",
    "AE",
    "CAE"
]