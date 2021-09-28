import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models, transforms


def init_weights(m):
  ''' Initialize network weights. Uses He initialization. 
  
  Usage: `net.apply(init_weights)`
  '''
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    init.kaiming_normal_(m.weight, nonlinearity='relu')


def load_backbone(unet):
  '''Initialize UNet encoder with VGG13 weights.
  
  Usage: `load_backbone(unet)`
  '''
  correspondence = [
    ('block1.first',  'features.0'),
    ('block1.second', 'features.2'),
    ('block2.first',  'features.5'),
    ('block2.second', 'features.7'),
    ('block3.first',  'features.10'),
    ('block3.second', 'features.12'),
    ('block4.first',  'features.15'),
    ('block4.second', 'features.17'),
    ('block5.first',  'features.20'),
    ('block5.second', 'features.22'),
  ]
  unet_weights = unet.state_dict()
  vgg_weights = models.vgg13(pretrained=True).state_dict()
  for u, v in correspondence:
    unet_weights[u+'.weight'] = vgg_weights[v+'.weight']
    unet_weights[u+'.bias'] = vgg_weights[v+'.bias']
  unet.load_state_dict(unet_weights)


## --- Model -------------------------------------------------------------------


class UNet(nn.Module):
  '''
  Input:  RGB image as array with shape (_, N_in, H, W), valued in [0, 1]. H and
          W must be multiples of 16. 
  Output: Class scores for each pixel.

  N_in:   number of input channels
  N_out:  number of output channels
  L:      number of latent channels (in first layer)
  '''
  def __init__(self, N_in=3, N_out=21, L=64):
    super().__init__()
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    self.block1 = DoubleConv(N_in,     L, (3, 3), padding='same')
    self.block2 = DoubleConv(   L,   2*L, (3, 3), padding='same')
    self.block3 = DoubleConv( 2*L,   4*L, (3, 3), padding='same')
    self.block4 = DoubleConv( 4*L,   8*L, (3, 3), padding='same')
    self.block5 = DoubleConv( 8*L,   8*L, (3, 3), padding='same')
    self.block6 = DoubleConv(16*L,   4*L, (3, 3), padding='same')
    self.block7 = DoubleConv( 8*L,   2*L, (3, 3), padding='same')
    self.block8 = DoubleConv( 4*L,     L, (3, 3), padding='same')
    self.block9 = DoubleConv( 2*L,     L, (3, 3), padding='same')
    self.out = nn.Conv2d(       L, N_out, 1)
  
  def forward(self, x):
    x = self.normalize(x)
    x_1 = self.block1(x)
    x_2 = nn.MaxPool2d(2)(x_1)
    x_2 = self.block2(x_2)
    x_3 = nn.MaxPool2d(2)(x_2)
    x_3 = self.block3(x_3)
    x_4 = nn.MaxPool2d(2)(x_3)
    x_4 = self.block4(x_4)
    x = nn.MaxPool2d(2)(x_4)
    x = self.block5(x)
    
    x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
    x = torch.cat((x_4, x), axis=1)
    x = self.block6(x)
    
    x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
    x = torch.cat((x_3, x), axis=1)
    x = self.block7(x)
    
    x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
    x = torch.cat((x_2, x), axis=1)
    x = self.block8(x)
    
    x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
    x = torch.cat((x_1, x), axis=1)
    x = self.block9(x)
    x = self.out(x)
    return x


class DoubleConv(nn.Module):
  ''' Convenience module with two Conv2D-ReLU layers. '''
  def __init__(self, in_channels, out_channels, kernel_sizes, **kwargs):
    super().__init__()
    self.first = nn.Conv2d(in_channels, out_channels, kernel_sizes[0], **kwargs)
    self.second = nn.Conv2d(out_channels, out_channels, kernel_sizes[1], **kwargs)
      
  def forward(self, x):
    x = self.first(x)
    x = nn.ReLU(inplace=True)(x)
    x = self.second(x)
    x = nn.ReLU(inplace=True)(x)
    return x
