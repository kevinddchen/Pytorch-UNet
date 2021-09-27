import torch


class Mean:
  '''Aggregate mean. Inputs are scalars.'''
  def __init__(self):
    self.s = torch.tensor(0).double().cuda()
    self.N = torch.tensor(0).double().cuda()

  def accumulate(self, x):
    self.s += x
    self.N += 1

  def result(self):
    return self.s / self.N


class PixelAccuracy:
  '''Pixel accuracy, taken over non-ignored classes. Inputs have shape=(N, H, W).'''
  def __init__(self, ignore_index=255):
    self.matches = torch.tensor(0).double().cuda()
    self.total = torch.tensor(0).double().cuda()
    self.ignore_index = ignore_index

  def accumulate(self, truth, pred):
    self.matches += (truth == pred).sum()
    self.total += (truth != self.ignore_index).sum()

  def result(self):
    return self.matches / self.total


class MeanIoU:
  '''Mean Intersection over Union, taken over non-ignored classes. Inputs have shape=(N, H, W).'''
  def __init__(self, num_classes=21, ignore_index=255):
    self.classes = torch.arange(num_classes).cuda()
    self.intersection = torch.zeros(num_classes).double().cuda()
    self.union = torch.zeros(num_classes).double().cuda()
    self.ignore_index = ignore_index

  def accumulate(self, truth, pred):
    pred[truth == self.ignore_index] = self.ignore_index      # ignored class
    pred_one_hot = pred.unsqueeze(-1) == self.classes
    truth_one_hot = truth.unsqueeze(-1) == self.classes
    self.intersection += (truth_one_hot & pred_one_hot).sum((0, 1, 2))
    self.union += (truth_one_hot | pred_one_hot).sum((0, 1, 2))

  def result(self):
    iou = self.intersection / self.union
    return iou.mean()
