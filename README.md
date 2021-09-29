# Pytorch-UNet

This is a PyTorch implementation of the UNet architecture outlined in <a href="https://arxiv.org/abs/1505.04597">Ronneberger et al. (2015)</a>, adapted to perform semantic image segmentation on the Pascal VOC dataset.
For a more detailed introduction, see https://github.com/kevinddchen/Keras-FCN.

## Install

To use out-of-the-box, run the following:

```
git clone git@github.com:kevinddchen/Pytorch-UNet.git
cd Pytorch-UNet
git lfs pull
```

Copy the images you want segmented into a new directory called `eval/images/`.
Create the conda environment in [environment.yml](environment.yml).
Then run

```
python eval.py
```

The labels will be saved to `eval/labels/`.

## Training

The model is in [model.py](model.py), and the training details are in [train.py](train.py).
We train on the 11,355 labelled images in the <a href="http://home.bharathh.info/pubs/codes/SBD/download.html">Berkeley Segmentation Boundaries Dataset (SBD)</a> and validate on the 676 labelled images in the original Pascal VOC dataset that are missing from the SBD.
If you want to duplicate our dataset, download the [val/](val) folder of this repository using the command `gif lfs pull`.
Then, download the SBD dataset from their website and place the contents of `benchmark_RELEASE/dataset/img/` into a new folder called `train/images/`, and `benchmark_RELEASE/dataset/cls/` into `train/labels/`.

The dataset is augmented by random scaling, rotation, cropping, and jittering of the RBG values.
Details are in [utils.py](utils.py#L206)
To train, run the command,

```
python train.py
```

The encoder weights are initialized with the VGG13 pretrained weights.
Training took 6 hours on a T4 GPU.
For comparison, we also trained a FCN using the VGG13 backbone, which has 10x the number of parameters.
That model can be found on the branch <a href="https://github.com/kevinddchen/Pytorch-UNet/tree/fcn">fcn</a>.

## Results

Below are some predicted labels in the validation set.

| Image | Truth | UNet | FCN |
| :--: | :--: | :--: | :--: |
| <img src="val/images/2007_000032.jpg" width=200> | <img src="val/labels/2007_000032.png" width=200> | <img src="samples/epoch20_0.png" width=200> | <img src="samples/fcn_epoch20_0.png" width=200> |
| <img src="val/images/2007_000033.jpg" width=200> | <img src="val/labels/2007_000033.png" width=200> | <img src="samples/epoch20_1.png" width=200> | <img src="samples/fcn_epoch20_1.png" width=200> |
| <img src="val/images/2007_000039.jpg" width=200> | <img src="val/labels/2007_000039.png" width=200> | <img src="samples/epoch20_2.png" width=200> | <img src="samples/fcn_epoch20_2.png" width=200> |
| <img src="val/images/2007_000042.jpg" width=200> | <img src="val/labels/2007_000042.png" width=200> | <img src="samples/epoch20_3.png" width=200> | <img src="samples/fcn_epoch20_3.png" width=200> |
| <img src="val/images/2007_000061.jpg" width=200> | <img src="val/labels/2007_000061.png" width=200> | <img src="samples/epoch20_4.png" width=200> | <img src="samples/fcn_epoch20_4.png" width=200> |
| <img src="val/images/2007_000063.jpg" width=200> | <img src="val/labels/2007_000063.png" width=200> | <img src="samples/epoch20_5.png" width=200> | <img src="samples/fcn_epoch20_5.png" width=200> |
| <img src="val/images/2007_000068.jpg" width=200> | <img src="val/labels/2007_000068.png" width=200> | <img src="samples/epoch20_6.png" width=200> | <img src="samples/fcn_epoch20_6.png" width=200> |
| <img src="val/images/2007_000121.jpg" width=200> | <img src="val/labels/2007_000121.png" width=200> | <img src="samples/epoch20_7.png" width=200> | <img src="samples/fcn_epoch20_7.png" width=200> |
| <img src="val/images/2007_000123.jpg" width=200> | <img src="val/labels/2007_000123.png" width=200> | <img src="samples/epoch20_8.png" width=200> | <img src="samples/fcn_epoch20_8.png" width=200> |
| <img src="val/images/2007_000129.jpg" width=200> | <img src="val/labels/2007_000129.png" width=200> | <img src="samples/epoch20_9.png" width=200> | <img src="samples/fcn_epoch20_9.png" width=200> |

The performance of these models on the validation set are summarized below.

| Model | UNet | FCN | 
| :--: | :--: | :--: | 
| Pixel accuracy | 0.878 | 0.903 | 
| Mean IoU | 0.490 | 0.583 |

<img src="assets/unet_stats.png" width=500>

<img src="assets/fcn_stats.png" width=500>
