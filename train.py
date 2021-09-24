import os
import datetime
import argparse
import cv2 as cv
import torch

from model import UNet, init_weights
from dataset import TrainDataset
import utils
from utils_time import TimeEstimator

if __name__ == '__main__':
  ## Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_image_dir', type=str, default='train/images', help='directory containing training images')
  parser.add_argument('--train_label_dir', type=str, default='train/labels', help='directory containing training labels')
  parser.add_argument('--val_image_dir', type=str, default='val/images', help='directory containing validation images')
  parser.add_argument('--val_label_dir', type=str, default='val/labels', help='directory containing validation labels')
  parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to save checkpoint models')
  parser.add_argument('--checkpoint_interval', type=int, default=1, help='save a checkpoint model every X epochs')
  parser.add_argument('--sample_dir', type=str, default='samples', help='directory to save validation samples')
  parser.add_argument('--sample_num', type=int, default=10, help='number of samples to save')
  parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='use cudnn benchmark')
  parser.add_argument('--use_amp', type=bool, default=True, help='use mixed precision')
  parser.add_argument('--num_workers', type=int, default=4, help='number of cpu workers for DataLoader')
  
  parser.add_argument('--epochs', type=int, default=40, help='')
  parser.add_argument('--batch_size', type=int, default=4, help='')
  parser.add_argument('--shuffle', type=bool, default=True, help='shuffle training data')
  parser.add_argument('--resume_epoch', type=int, default=0, help='if non-zero, resume training from X epoch')
  parser.add_argument('--lr', type=float, default=1e-4, help='Adam: learning rate')
  parser.add_argument('--b1', type=float, default=0.9, help='Adam: beta 1')
  parser.add_argument('--b2', type=float, default=0.999, help='Adam: beta 2')
  parser.add_argument('--weight_decay', type=float, default=0, help='Adam: weight decay')
  parser.add_argument('--lr_decay_step', type=int, default=10, help='decay learning rate every X epochs')
  parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='factor to decay learning rate')
  opt = parser.parse_args()

  ## initialize
  os.makedirs(opt.checkpoint_dir, exist_ok=True)
  os.makedirs(opt.sample_dir, exist_ok=True)

  if not torch.cuda.is_available():
    print("Error: training on cpu is not permitted.")
    exit(1)
  
  torch.backends.cudnn.benchmark = opt.cudnn_benchmark

  ## ---------------------------------------------------------------------------
  
  ## load model
  net = UNet()
  if opt.resume_epoch:
    print("Resuming from epoch: {}".format(opt.resume_epoch))
    filename = os.path.join(opt.checkpoint_dir, utils.checkpoint_name(opt.resume_epoch))
    pretrained_dict = torch.load(filename)
    net.load_state_dict(pretrained_dict)
  else:
    print("Initializing random weights")
    net.apply(init_weights)
  net = net.cuda()

  ## load data
  trainset = TrainDataset(opt.train_image_dir, opt.train_label_dir)
  trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=opt.batch_size, 
    shuffle=opt.shuffle, 
    num_workers=opt.num_workers, 
    pin_memory=True,
    drop_last=True
  )
  valset = TrainDataset(opt.val_image_dir, opt.val_label_dir)
  valloader = torch.utils.data.DataLoader(
    valset, 
    num_workers=opt.num_workers, 
    pin_memory=True
  )
  print("Training batch size: {}".format(opt.batch_size))
  print("Number of training batches: {}".format(len(trainloader)))
  print("Number of validation images: {}".format(len(valloader)))

  ## loss
  CrossEntropyLoss = torch.nn.CrossEntropyLoss(ignore_index=255)

  ## optimizer
  optimizer = torch.optim.Adam(
    net.parameters(), 
    lr=opt.lr, 
    betas=(opt.b1, opt.b2), 
    weight_decay=opt.weight_decay
  )

  ## learning rate decay
  scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=opt.lr_decay_step,
    gamma=opt.lr_decay_factor,
    last_epoch=opt.resume_epoch-1
  )

  ## mixed precision
  scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)

  timeEstimator = TimeEstimator((opt.epochs - opt.resume_epoch) * (len(trainloader) + len(valloader)))

  ## ---------------------------------------------------------------------------

  for epoch in range(opt.resume_epoch, opt.epochs):

    ##  === train ===
    net.train()
    for batch_i, (image, label) in enumerate(trainloader):
      image = image.cuda()
      label = label.cuda()
      optimizer.zero_grad(set_to_none=True)

      with torch.cuda.amp.autocast(enabled=opt.use_amp):
        out = net(image)
        loss = CrossEntropyLoss(out, label)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      delta_t, remaining_t = timeEstimator.update()
      print("TRAIN | Epoch {}/{} | Batch {}/{} | Loss {:.3f} | {:.3f} sec | {} remaining".format(
        epoch+1, opt.epochs, batch_i+1, len(trainloader), loss.item(), delta_t, datetime.timedelta(seconds=remaining_t)
      ))
    

    ## === validate ===
    net.eval()
    for batch_i, (image, label) in enumerate(valloader):
      image = image.cuda()
      label = label.cuda()

      with torch.no_grad():
        out = net(image)
        pred = torch.argmax(out, 1)
        loss = CrossEntropyLoss(out, label)

      ## save label as image
      if batch_i < opt.sample_num:
        pred = pred[0].cpu().numpy()
        pred_img = utils.label_to_image(pred)
        filename = os.path.join(opt.sample_dir, utils.sample_name(epoch+1, batch_i))
        utils.save_image(filename, pred_img)

      delta_t, remaining_t = timeEstimator.update(count_toward_average=False)
      print("VAL-- | Epoch {}/{} | Batch {}/{} | Loss {:.3f} | {:.3f} sec | {} remaining".format(
        epoch+1, opt.epochs, batch_i+1, len(valloader), loss.item(), delta_t, datetime.timedelta(seconds=remaining_t)
      ))
    

    ## === save checkpoint ===
    if (epoch + 1) % opt.checkpoint_interval == 0:
      filename = os.path.join(opt.checkpoint_dir, utils.checkpoint_name(epoch+1))
      torch.save(net.state_dict(), filename)

    ## === adjust learning rate ===
    scheduler.step()