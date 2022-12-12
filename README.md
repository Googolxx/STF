# The Devil Is in the Details: Window-based Attention for Image Compression
Pytorch implementation of the paper "The Devil Is in the Details: Window-based Attention for Image Compression". CVPR2022.
This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI). We kept scripts for training and evaluation, and removed other components. The major changes are provided in `compressai/models`. For the official code release, see the [CompressAI](https://github.com/InterDigitalInc/CompressAI).

## About
This repo defines the CNN-based models and Transformer-based models for learned image compression in "The Devil Is in the Details: Window-based Attention for Image Compression".


![cnn_arch](https://github.com/Googolxx/STF/blob/main/assets/cnn_arch.png)
>  The architecture of CNN-based model.

![stf_arch](https://github.com/Googolxx/STF/blob/main/assets/stf_arch.png)
>  The architecture of Transformer-based model (STF).


## Installation

Install [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the packages required for development.
```bash
conda create -n compress python=3.7
conda activate compress
pip install compressai
pip install pybind11
git clone https://github.com/Googolxx/STF stf
cd stf
pip install -e .
pip install -e '.[dev]'
```

> **Note**: wheels are available for Linux and MacOS.

## Usage

### Training
An examplary training script with a rate-distortion loss is provided in
`train.py`. 

Training a CNN-based model:
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py -d /path/to/image/dataset/ -e 1000 --batch-size 16 --save --save_path /path/to/save/ -m cnn --cuda --lambda 0.0035
e.g., CUDA_VISIBLE_DEVICES=0,1 python train.py -d openimages -e 1000 --batch-size 16 --save --save_path ckpt/cnn_0035.pth.tar -m cnn --cuda --lambda 0.0035
```
Training a Transformer-based model(STF):
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py -d /path/to/image/dataset/ -e 1000 --batch-size 16 --save --save_path /path/to/save/ -m stf --cuda --lambda 0.0035
```


### Evaluation

To evaluate a trained model on your own dataset, the evaluation script is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m compressai.utils.eval_model -d /path/to/image/folder/ -r /path/to/reconstruction/folder/ -a stf -p /path/to/checkpoint/ --cuda
```
```bash
CUDA_VISIBLE_DEVICES=0 python -m compressai.utils.eval_model -d /path/to/image/folder/ -r /path/to/reconstruction/folder/ -a cnn -p /path/to/checkpoint/ --cuda
```


### Dataset
The script for downloading [OpenImages](https://github.com/openimages) is provided in `downloader_openimages.py`. Please install [fiftyone](https://github.com/voxel51/fiftyone) first.

## Results

### Visualization

![visualization01](https://github.com/Googolxx/STF/blob/main/assets/detail_01.png)
>  Visualization of the reconstructed image kodim01.png.

![visualization07](https://github.com/Googolxx/STF/blob/main/assets/detail_07.png)
>  Visualization of the reconstructed image kodim07.png.
>
### RD curves

![kodak_rd](https://github.com/Googolxx/STF/blob/main/assets/kodak_rd.png)
>  RD curves on [Kodak](http://r0k.us/graphics/kodak/).

![clic_rd](https://github.com/Googolxx/STF/blob/main/assets/clic_rd.png)
>  RD curves on [CLIC Professional Validation dataset](https://www.compression.cc/).

### Codec Efficiency on [Kodak](http://r0k.us/graphics/kodak/)
| Method | Enc(s) | Dec(s) | PSNR | bpp |
| ------------ | ------ | ------ | ------ | ------ |
| CNN | 0.12 | 0.12 | 35.91 | 0.650 |
| STF | 0.15 | 0.15 | 35.82 | 0.651 |

### Pretrained Models
Pretrained models (optimized for MSE) trained from scratch using randomly chose 300k images from the OpenImages dataset.

| Method | Lambda | Link                                                                                              |
| ---- |--------|---------------------------------------------------------------------------------------------------|
| CNN | 0.0035 | [cnn_0035](https://drive.google.com/file/d/1VUV6_Ws-3X6VZDKkfOVPQel8nr7ecut9/view?usp=sharing)    |
| CNN | 0.025  | [cnn_025](https://drive.google.com/file/d/1LrAWPlBE6WJUfjiDPGFO8ANSaP5BFEQI/view?usp=sharing)     |
| STF | 0.0018 | [stf_0018](https://drive.google.com/file/d/1ul3RPXxPbvE_2UkdyZZ0r7tM2Kevg-3F/view?usp=share_link) |
| STF | 0.0035 | [stf_0035](https://drive.google.com/file/d/1OFzZoEaofNgsimBuOPHtgOJiGsR_RS-M/view?usp=sharing)    |
| STF | 0.0067 | [stf_0067](https://drive.google.com/file/d/1SjhqcKyP3SqVm4yhJQslJ6HgY1E8FcBL/view?usp=share_link) |
| STF | 0.013  | [stf_013](https://drive.google.com/file/d/1mupv4vcs8wpNdXCPclXghliikJyYjgj-/view?usp=share_link)  |
| STF | 0.025  | [stf_025](https://drive.google.com/file/d/1rsYgEYuqSYBIA4rfvAjXtVSrjXOzkJlB/view?usp=sharing)     |
| STF | 0.0483 | [stf_0483](https://drive.google.com/file/d/1cH5cR-0VdsQqCchyN3DO62Sx0WGjv1h8/view?usp=share_link)    |

Other pretrained models will be released successively.
## Citation
```
@inproceedings{zou2022the,
  title={The Devil Is in the Details: Window-based Attention for Image Compression},
  author={Zou, Renjie and Song, Chunfeng and Zhang, Zhaoxiang},
  booktitle={CVPR},
  year={2022}
}
```

## Related links
 * CompressAI: https://github.com/InterDigitalInc/CompressAI
 * Swin-Transformer: https://github.com/microsoft/Swin-Transformer
 * Tensorflow compression library by Ballé et al.: https://github.com/tensorflow/compression
 * Range Asymmetric Numeral System code from Fabian 'ryg' Giesen: https://github.com/rygorous/ryg_rans
 * Kodak Images Dataset: http://r0k.us/graphics/kodak/
 * Open Images Dataset: https://github.com/openimages
 * fiftyone: https://github.com/voxel51/fiftyone
 * CLIC: https://www.compression.cc/


