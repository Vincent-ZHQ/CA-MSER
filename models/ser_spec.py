import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


'''
2 Models Available:
   - SER_AlexNet     : AlexNet model from pyTorch (CNN features layer + FC classifier layer)
'''



class SER_AlexNet(nn.Module):
    """
    Reference:
    https://pytorch.org/docs/stable/torchvision/models.html#id1

    AlexNet model from torchvision package. The model architecture is slightly
    different from the original model.
    See: AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.


    Parameters
    ----------
    num_classes : int
    in_ch   : int
        The number of input channel.
        Default AlexNet input channels is 3. Set this parameters for different
            numbers of input channels.
    pretrained  : bool
        To initialize the weight of AlexNet.
        Set to 'True' for AlexNet pre-trained weights.

    Input
    -----
    Input dimension (N,C,H,W)

    N   : batch size
    C   : channels
    H   : Height
    W   : Width

    Output
    ------
    logits (before Softmax)

    """


    def __init__(self,num_classes=4, in_ch=3, pretrained=True):
        super(SER_AlexNet, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool
        self.classifier = model.classifier

        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            init_layer(self.features[0])

        self.classifier[6] = nn.Linear(4096, num_classes)

        self._init_weights(pretrained=pretrained)
        
        print('\n<< SER AlexNet Finetuning model initialized >>\n')

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x_ = torch.flatten(x, 1)
        out = self.classifier(x_)

        return x, out

    def _init_weights(self, pretrained=True):

        init_layer(self.classifier[6])

        if pretrained == False:
            init_layer(self.features[0])
            init_layer(self.features[3])
            init_layer(self.features[6])
            init_layer(self.features[8])
            init_layer(self.features[10])
            init_layer(self.classifier[1])
            init_layer(self.classifier[4])
