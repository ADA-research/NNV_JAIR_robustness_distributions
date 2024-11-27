# Code for Robustness Distributions in Neural Network Verification

This repository contains the code and models from the paper:  
**Robustness Distributionsin Neural Network Verification**  
*Author(s): [Your Name(s)]*  
Published in: [Conference/Journal Name, Year]  

---

## Table of Contents
- [Code for Robustness Distributions in Neural Network Verification](#code-for-robustness-distributions-in-neural-network-verification)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository Structure](#repository-structure)
  - [External Packages](#external-packages)

---

## Overview
This repository provides:
- Pre-trained models in **ONNX** format.
- PyTorch implementations for training and verification of the models.
- Experimentation scripts and instructions for reproducing results on **MNIST**, **CIFAR-10**, and **GTSRB** datasets.

The aim of this project is to [briefly describe the main objective/purpose of your research or code].

---

## Repository Structure
- network_onnx: Network in the onnx formats
  - [MNIST](networks_onnx/mnist)
    - [mnist_6_256_fgsm.onnx](networks_onnx/mnist/mnist_6_256_fgsm.onnx)
    - [mnist_6_256_pgd.onnx](networks_onnx/mnist/mnist_6_256_pgd.onnx)
    - [mnist_6_256_standard.onnx](networks_onnx/mnist/mnist_6_256_standard.onnx)
    - [mnist_7_1024_fgsm.onnx](networks_onnx/mnist/mnist_7_1024_fgsm.onnx)
    - [mnist_7_1024_pgd.onnx](networks_onnx/mnist/mnist_7_1024_pgd.onnx)
    - [mnist_7_1024_standard.onnx](networks_onnx/mnist/mnist_7_1024_standard.onnx)
  - [CIFAR-10](networks_onnx/cifar-10)
    - [cifar_7_1024_fgsm.onnx](networks_onnx/cifar-10/cifar_7_1024_fgsm.onnx)
    - [cifar_7_1024_pgd.onnx](networks_onnx/cifar-10/cifar_7_1024_pgd.onnx)
    - [cifar_7_1024_standard.onnx](networks_onnx/cifar-10/cifar_7_1024_standard.onnx)
    - [conv_big_fgsm.onnx](networks_onnx/cifar-10/conv_big_fgsm.onnx)
    - [conv_big_pgd.onnx](networks_onnx/cifar-10/conv_big_pgd.onnx)
    - [conv_big_standard.onnx](networks_onnx/cifar-10/conv_big_standard.onnx)
    - [resnet_4b_fgsm.onnx](networks_onnx/cifar-10/resnet_4b_fgsm.onnx)
    - [resnet_4b_pgd.onnx](networks_onnx/cifar-10/resnet_4b_pgd.onnx)
    - [resnet_4b_standard.onnx](networks_onnx/cifar-10/resnet_4b_standard.onnx)
  - [GTSRB](networks_onnx/gtsrb)
    - [gtsrb_6_256_fgsm.onnx](networks_onnx/gtsrb/gtsrb_6_256_fgsm.onnx)
    - [gtsrb_6_256_pgd.onnx](networks_onnx/gtsrb/gtsrb_6_256_pgd.onnx)
    - [gtsrb_6_256_standard.onnx](networks_onnx/gtsrb/gtsrb_6_256_standard.onnx)
    - [gtsrb_7_1024_fgsm.onnx](networks_onnx/gtsrb/gtsrb_7_1024_fgsm.onnx)
    - [gtsrb_7_1024_pgd.onnx](networks_onnx/gtsrb/gtsrb_7_1024_pgd.onnx)
    - [gtsrb_7_1024_standard.onnx](networks_onnx/gtsrb/gtsrb_7_1024_standard.onnx)
- training: Code for the training and PyTorch implementations for the networks
- verification: Code for verifying the networks
  
## External Packages
This project uses the following external packages:
- [VERONA](https://github.com/ADA-research/VERONA): An open-source package for creating Robustness Distributions.
- [adversarial-training-box](https://github.com/Aaron99B/adversarial-training-box): An open-source package for adversarial training of neural networks with PyTorch.