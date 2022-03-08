# The Devil Is in the Details: Window-based Attention for Image Compression
Pytorch implementation of the paper "The Devil Is in the Details: Window-based Attention for Image Compression" by Zou et.al..

This repository currently provides the code for training and evaluating the CNN-based models and Transformer-based models.

The code is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI). For the official code release, see the [CompressAI](https://github.com/InterDigitalInc/CompressAI).


## Abstract
Learned image compression methods have exhibited superior rate-distortion performance than classical image compression standards.
Most existing learned image compression models are based on Convolutional Neural Networks (CNNs).
Despite great contributions, a main drawback of CNN based model is that its structure is not designed for capturing local details, especially the nonrepetitive textures, which severely affects the reconstruction quality.
Therefore, how to make full use of both global structure and local texture becomes the core problem for learning-based image compression.
Inspired by recent progresses of Vision Transformer (ViT) and Swin Transformer, we found that combining the local-aware attention mechanism with the global-related feature learning could meet the expectation in image compression.
In this paper, we first extensively study the effects of multiple kinds of attention mechanisms for local features learning, then introduce a more straightforward yet effective window-based local attention block.
The proposed window-based attention is very flexible which could work as a plug-and-play component to enhance CNN and Transformer models.
Moreover, we propose a novel Symmetrical TransFormer (STF) framework with absolute transformer blocks in the down-sampling encoder and up-sampling decoder, which may be the first exploration of designing the up-sampling transformer, especially for the image compression task.
Extensive experimental evaluations have shown that the proposed method is effective and outperforms the state-of-the-art methods.

![guess](https://github.com/Googolxx/STF/blob/main/assets/cnn_arch.pdf)

## Installation

Install [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the packages required for development.
```bash
conda create -n compress python=3.7
conda activate compress
pip install compressai
cd /home/renjie_zou/release/ImageCompression
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
```
Training a Transformer-based model(STF):
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py -d /path/to/image/dataset/ -e 1000 --batch-size 16 --save --save_path /path/to/save/ -m stf --cuda --lambda 0.0035
```


### Evaluation

To evaluate a trained model on your own dataset, the evaluation script is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m compressai.utils.eval_model -d kodak -r kodak_reconstruction -a stf -p /path/to/checkpoint/ --cuda
```
```bash
CUDA_VISIBLE_DEVICES=0 python -m compressai.utils.eval_model -d kodak -r kodak_reconstruction -a cnn -p /path/to/checkpoint/ --cuda
```

### Results

To be continued...

![RD curves](results/full_rd.jpg)

## Related links
 * Tensorflow compression library by Ballé et al.: https://github.com/tensorflow/compression
 * Range Asymmetric Numeral System code from Fabian 'ryg' Giesen: https://github.com/rygorous/ryg_rans
 * BPG image format by Fabrice Bellard: https://bellard.org/bpg
 * HEVC HM reference software: https://hevc.hhi.fraunhofer.de
 * VVC VTM reference software: https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM
 * AOM AV1 reference software: https://aomedia.googlesource.com/aom
 * Z. Cheng et al. 2020: https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention
 * Kodak image dataset: http://r0k.us/graphics/kodak/
 * CompressAI: https://github.com/InterDigitalInc/CompressAI

