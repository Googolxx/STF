# The Devil Is in the Details: Window-based Attention for Image Compression


## Installation

Install CompressAI https://github.com/InterDigitalInc/CompressAI and the packages required for development.
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

### Examples

Script and notebook examples can be found in the `examples/` directory.


#### Training a model
An examplary training script with a rate-distortion loss is provided in
`examples/train.py`. 

Training a CNN model:
```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/train.py -d /data2/renjie_zou/openimages_30 -e 1000 --batch-size 16 --save --save_path ckpt/test.pth.tar -m ccwo --cuda --lambda 0.013
```
Training a Transformer model:
```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/train.py -d /data2/renjie_zou/openimages_30 -e 1000 --batch-size 16 --save --save_path ckpt/test.pth.tar -m stf --cuda --lambda 0.013
```


#### Evaluation

To evaluate a trained model on your own dataset, the evaluation script is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m compressai.utils.eval_model checkpoint  kodak kodak_r -a stf -p /home/renjie_zou/compressai/ckpt/cc_w_o_013_best.pth.tar --cuda
```
```bash
CUDA_VISIBLE_DEVICES=0 python -m compressai.utils.eval_model -d kodak -r kodak_r -a cnn -p /home/renjie_zou/compressai/ckpt/cc_w_o_013_best.pth.tar --cuda
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

