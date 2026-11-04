# Unsupervised Domain Adaptive Thermal Infrared Tracking (IEEE TMM 2025)

## Abstract
We propose an unsupervised Dual-level Domain Adaptation TIR Tracking framework (DDAT), which can benefit from training on large-scale labeled RGB
datasets and unlabeled TIR datasets. Specifically, to transfer the useful knowledge learned from the RGB dataset to TIR tracking, we first propose an adversarial-based adaptation module
on both the semantic-level and the feature-level. While the semantic-level adaptation can reduce the semantic gap between
the TIR and RGB tracking tasks, the feature-level adaptation can learn domain-invariant features for more robust tracking.
Second, we propose a partial domain adaptation module to alleviate the negative transfer problem because the RGB and
TIR tracking domains have non-identical class and feature spaces. Instead of aligning the entire feature space, this module adaptively selects partial similarity samples and features for
alignment, thus obtaining more fine-grained aligned results. Third, we collect the currently largest-scale unlabeled TIR dataset to
train the proposed framework.

<img width="1521" height="633" alt="0a1e0bb5-89d3-484b-ac7e-86f668e506b6" src="https://github.com/user-attachments/assets/0850ad9f-5561-46a3-a9e5-5d6959bcbb3b" />

## Download
- You can download our trained models from [Baidu Pan](https://pan.baidu.com/s/1lj-rMX_6lXj5jvyrm3g8DQ?pwd=1111). Extraction Code: **1111**

- We provide a raw result of DDAT on the LSOTB-TIR100, LSOTB-TIR120, and PTB-TIR benchmarks in [here](https://pan.baidu.com/s/1-r5-W6TgNlWcGFwFEYzB3A?pwd=1111). Extraction Code: **1111**

## Usage

### Tracking
- Clone the code and unzip it on your computer.
- Prerequisites: Ubuntu 22.04, Pytorch 2.2.2, GTX A100, CUDA 12.1.
- Download our trained models from [here](https://pan.baidu.com/s/1-r5-W6TgNlWcGFwFEYzB3A?pwd=1111).
- Run `pysot_toolkit/test.py` to test a TIR sequence using the default model.

## Citation
If you use the code or dataset, please consider citing our paper.

## Contact
Feedback and comments are welcome!  
Feel free to contact us via **liuqiao.hit@gmail.com** or **liuqiao@stu.hit.edu.cn**.
