import os
from torch.utils.data import Dataset

import utils


## --- for training ------------------------------------------------------------


class TrainDataset(Dataset):
  def __init__(self, image_dir, label_dir):
    self.image_dir = image_dir
    self.label_dir = label_dir
    self.image_list = sorted(utils.get_names(image_dir))
    self.label_list = sorted(utils.get_names(label_dir))
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
    image, label = utils.resize_for_train(image, label)
    return image, label


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
    image = utils.image_to_tensor(image)
    image = utils.resize_for_eval(image)
    name = self.image_list[idx].split('.')[0]   # remove extension
    return image, name
