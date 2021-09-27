import os
import datetime
import argparse
import torch

from model import UNet
from dataset import EvalDataset
import utils
from utils_time import TimeEstimator

if __name__ == '__main__':
  ## Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--pretrained_weights', '-w', type=str, required=True, help='path to pretrained model weights')
  parser.add_argument('--input_image_dir', type=str, default='eval/images', help='directory containing images to segment')
  parser.add_argument('--output_label_dir', type=str, default='eval/labels', help='directory to save segments')
  parser.add_argument('--num_workers', type=int, default=4, help='(only for gpu) number of cpu workers for DataLoader')
  opt = parser.parse_args()

  ## initialize
  os.makedirs(opt.output_label_dir, exist_ok=True)

  use_cuda = torch.cuda.is_available()
  if use_cuda:
      device = torch.device('cuda')
  else:
      device = torch.device('cpu')
      opt.num_workers = 0
  print("Device: {}".format(device))

  ## ---------------------------------------------------------------------------
  
  ## load model
  net = UNet()
  pretrained_dict = torch.load(opt.pretrained_weights, map_location=device)
  net.load_state_dict(pretrained_dict)
  net = net.to(device)

  ## load data
  evalset = EvalDataset(opt.input_image_dir)
  evalloader = torch.utils.data.DataLoader(evalset, num_workers=opt.num_workers, pin_memory=use_cuda)
  print("Number of evaluation images: {}".format(len(evalloader)))

  timeEstimator = TimeEstimator(len(evalloader))

  ## ---------------------------------------------------------------------------

  ## === evaluate ===
  net.eval()
  for batch_i, (image, name) in enumerate(evalloader):
    image = image.to(device)

    with torch.no_grad():
      out = net(image)
    
    label = out.argmax(1)

    ## save label as image
    ## TODO: save label in original dimensions as palettised png, or other convenient form
    label = label[0].cpu().numpy()
    label_img = utils.label_to_image(label)
    filename = os.path.join(opt.output_label_dir, name[0]+'.png')
    utils.save_image(filename, label_img)

    delta_t, remaining_t = timeEstimator.update()
    print("EVAL | Batch {}/{} | {:.2f} sec | {} remaining".format(
      batch_i+1, len(evalloader), delta_t, datetime.timedelta(seconds=remaining_t)
    ))

  
  total_t = timeEstimator.total()
  print("Total Elapsed Time: {}".format(datetime.timedelta(seconds=total_t)))
