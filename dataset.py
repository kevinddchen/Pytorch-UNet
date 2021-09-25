import os
import numpy as np
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset

import utils


## --- for training ------------------------------------------------------------


class TrainDataset(Dataset):
  def __init__(self, image_dir, label_dir):
    self.image_dir = image_dir
    self.label_dir = label_dir
    self.image_list = sorted(utils.get_names(image_dir))
    self.label_list = sorted(utils.get_names(label_dir))
    self.rng = np.random.default_rng()
    ## check filenames are consistent and label extensions are supported
    for i, i_name in enumerate(self.image_list):
      i_pre, _ = i_name.split('.')
      l_pre, l_ext = self.label_list[i].split('.')
      assert i_pre == l_pre, "Image and label names do not match: {}, {}".format(i_pre, l_pre)
      assert l_ext in ['png', 'mat'], "Extension not supported: {}".format(l_ext)

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    image_path = os.path.join(self.image_dir, self.image_list[idx])
    image = utils.get_image(image_path)

    label_path = os.path.join(self.label_dir, self.label_list[idx])
    ext = label_path.split('.')[-1]
    if ext == 'mat':
      label = utils.get_label_mat(label_path)
    elif ext == 'png':
      label = utils.get_label_png(label_path)
    else:
      assert 0, "Extension not supported: {}".format(ext)

    image = utils.image_to_tensor(image)
    label = utils.label_to_tensor(label)
    image, label = self.preprocess(image, label)
    return image, label

  def preprocess(self, image, label, target=512):
    '''Dataset augmentation with small rotations, scalings, and jitterings.'''
    h, w = label.shape
    label = label.unsqueeze(0)  # needed for many of these operations

    ## do `utils.resize_for_train`
    scale_factor = max(h, w) / float(target)
    new_w = (int(w / scale_factor) // 16) * 16
    new_h = (int(h / scale_factor) // 16) * 16
    image = TF.resize(image, (new_h, new_w))
    label = TF.resize(label, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)

    ## center pad
    l_pad, t_pad = (target-new_w)//2, (target-new_h)//2
    r_pad, b_pad = target-l_pad-new_w, target-t_pad-new_h
    image = TF.pad(image, (l_pad, t_pad, r_pad, b_pad))
    label = TF.pad(label, (l_pad, t_pad, r_pad, b_pad), fill=255)

    ## apply random affine transformation
    angle = self.rng.uniform(-5, 5)   # max 5 degrees either way
    scale = self.rng.uniform(1, 2)    # max x2 zoom
    x_max_trans = max(0, int(new_w*scale - target)) // 2
    y_max_trans = max(0, int(new_h*scale - target)) // 2
    x_trans = self.rng.integers(-x_max_trans, x_max_trans, endpoint=True)
    y_trans = self.rng.integers(-y_max_trans, y_max_trans, endpoint=True)
    image = TF.affine(image, angle, (x_trans, y_trans), scale, 0)
    label = TF.affine(label, angle, (x_trans, y_trans), scale, 0)

    ## adjust brightness and hue
    bright = self.rng.uniform(0.8, 1.2)
    hue = self.rng.uniform(-0.05, 0.05)
    image = TF.adjust_brightness(image, bright)
    image = TF.adjust_hue(image, hue)

    return image, label.squeeze()


## --- for evaluation ----------------------------------------------------------


class EvalDataset(Dataset):
  def __init__(self, image_dir):
    self.image_dir = image_dir
    self.image_list = sorted(utils.get_names(image_dir))
  
  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    image_path = os.path.join(self.image_dir, self.image_list[idx])
    image = utils.get_image(image_path)
    image = utils.resize_for_eval(image)
    image = utils.image_to_tensor(image)
    name = self.image_list[idx].split('.')[0]   # remove extension
    return image, name
