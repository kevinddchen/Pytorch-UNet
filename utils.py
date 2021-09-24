import os
import torch
import numpy as np
import cv2 as cv
import scipy.io # to read .mat files
from PIL import Image # to read image files


## --- Input/output ------------------------------------------------------------


def get_image(path):
  '''Retrieve image as array of RGB values from .jpg file.
  
  Args:
    path (string): Path to .jpg file
      
  Returns:
    (array<np.uint8>): RGB image. Shape=(height, width, 3)
  '''
  jpg = cv.imread(path)
  jpg = cv.cvtColor(jpg, cv.COLOR_BGR2RGB)
  return jpg


def get_label_mat(path):
  '''Retrieve class labels for each pixel from Berkeley SBD .mat file.
  
  Args:
    path (string): Path to .mat file
  
  Returns:
    (array<np.uint8>): Class as an integer in [0, 20] for each pixel. Shape=(height, width)
  '''
  mat = scipy.io.loadmat(path)
  arr = mat['GTcls']['Segmentation'].item(0,0) # this is how segmentation is stored
  return arr


def get_label_png(path):
  '''Retrieve class labels for each pixel from Pascal VOC .png file.
  
  Args:
    path (string): Path to .png file
  
  Returns:
    (array<np.uint8>): Class as an integer in [-1, 20], where -1 is boundary, for each pixel. Shape=(height, width)
  '''
  png = Image.open(path) # image is saved as palettised png. OpenCV cannot load without converting.
  arr = np.array(png)
  return arr


def save_image(filename, image):
  '''Save RGB image.
  
  Args:
    image (array<np.uint8>): RGB image. Shape=(height, width, 3)
    filename (string): path to file
  '''
  image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
  cv.imwrite(filename, image)


## --- Conversions -------------------------------------------------------------


PALETTE = np.reshape([
  0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128,
  128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0,
  128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0,
  192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
  64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128,
  64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128,
  64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0,
  64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192,
  128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192,
  128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192,
  64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192,
  32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32,
  128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96,
  0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0,
  32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192,
  128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64,
  128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160,
  128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64,
  224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128,
  192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32,
  64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64,
  64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192,
  224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128,
  128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160,
  0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0,
  96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0,
  224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0,
  64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32,
  64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128,
  160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192,
  192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224,
  64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64,
  96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192,
  64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160,
  0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224,
  32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128,
  224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128,
  160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224,
  0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32,
  32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192,
  32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224,
  160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96,
  64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32,
  224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224,
  64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192], (-1, 3))


def label_to_image(label, palette=PALETTE):
  '''Converts class labels to color image using a palette.
  
  Args:
    label (array<np.uint8>): Class label for each pixel. Shape=(height, width)
    palette (array<np.uint8>): RGB values for each class. Shape=(255, 3)
      
  Returns:
    (array<np.uint8>): RGB image. Shape=(height, width, 3)
  '''
  return palette[label].astype(np.uint8)


def image_to_tensor(image):
  '''Convert RBG image into pytorch tensor for neural network input.

  Args:
    image (array<np.uint8>): RGB image. Shape=(height, width, 3)
  
  Returns:
    (tensor<torch.float32>): Valued in (0, 1). Shape=(3, height, width)
  '''
  return torch.from_numpy(image.astype(np.float32) / 255.).permute(2, 0, 1).contiguous()


def label_to_tensor(label):
  '''Convert RBG image into pytorch tensor for neural network input.

  Args:
    image (array<np.uint8>): Class label for each pixel. Shape=(height, width)
  
  Returns:
    (tensor<torch.long>): Shape=(height, width)
  '''
  return torch.from_numpy(label.astype(np.int64))


## --- Resizing ----------------------------------------------------------------


def resize_for_train(image, label, target=512):
  '''Resize image and label to target resolution while preserving aspect ratio,
  padding if necessary. Assumes that image and label have the same dimensions.

  Args:
    image (array<np.uint8>): RGB image. Shape=(height, width, 3)
    label (array<np.uint8>): Class label for each pixel. Shape=(height, width)
    target: dimension of output image, in pixels

  Returns:
    (array<np.uint8>): Resized image. Shape=(target, target, 3)
    (array<np.uint8>): Resized label. Shape=(target, target)
  '''
  h, w, _ = image.shape
  scale_factor = max(h, w) / float(target)
  new_w = (int(w / scale_factor) // 16) * 16
  new_h = (int(h / scale_factor) // 16) * 16
  new_image = cv.resize(image, (new_w, new_h), None, 0, 0, cv.INTER_LINEAR)
  new_image = np.pad(new_image, ((0, target-new_h), (0, target-new_w), (0, 0)), 'constant', constant_values=0)
  new_label = cv.resize(label, (new_w, new_h), None, 0, 0, cv.INTER_NEAREST)
  new_label = np.pad(new_label, ((0, target-new_h), (0, target-new_w)), 'constant', constant_values=255)
  return new_image, new_label


def resize_for_eval(image, target=512):
  '''Resize image so that largest dimension equals `target` and both dimensions
  are divisible by 16. Aspect ratio will be preserved as best as possible.

  Args:
    image (array<np.uint8>): RGB image. Shape=(height, width, 3)
    target: largest dimension of output image, in pixels

  Returns:
    (array<np.uint8>): Resized RGB image. Shape=(new_height, new_width, 3)
  '''
  h, w, _ = image.shape
  scale_factor = max(h, w) / float(target)
  new_w = (int(w / scale_factor) // 16) * 16
  new_h = (int(h / scale_factor) // 16) * 16
  new_image = cv.resize(image, (new_w, new_h), None, 0, 0, cv.INTER_LINEAR)
  return new_image


## --- Misc --------------------------------------------------------------------


def get_names(path):
  # read a folder, return all filenames
  f = []
  for root, dirs, files in os.walk(path):
    f.extend(files)
  return f


def sample_name(epoch, i):
  # name for sample labels
  return "epoch{}_{}.png".format(epoch, i)


def checkpoint_name(epoch):
  # name for checkpoint weights
  return "unet_epoch{}.pth".format(epoch)
