# Code for Robustness Distributions in Neural Network Verification

This repository contains the code and models from the paper:  

**Robustness Distributions in Neural Network Verification**  

*Author(s): Annelot W. Bosman, Aaron Berger Holger H. Hoos and Jan N. van Rijn \
Published in: Under review at JAIR.

Please use this citation key when using any of the information from this repository:

```
@inproceedings{BosEtAl23,
    author = {Bosman, Annelot W. and Hoos, Holger H. and van Rijn, Jan N.},
    title = {A Preliminary Study of Critical Robustness Distributions in Neural Network Verification},
    year = {2023},
    booktitle = {Workshop on Formal Methods for ML-Enabled Autonomous Systems (FOMLAS)},
    volume = {6}}
```


---


## Overview
This repository provides:
- Pre-trained models in **ONNX** format.
- PyTorch implementations for training and verification of the models.
- Experimentation scripts and instructions for reproducing results on **MNIST**, **CIFAR-10**, and **GTSRB** datasets.
- Data used for all figures and tables in the JAIR paper.

------

## Repository Structure
- **data**: Folder containing the results from our verification experiments. Csv-files with the structure ```results_{dataset}_{paper}.csv``` contain the lower and upper bound for the robustness distribution for each separate image on both training and testing set for each investigated network. Csv-files with the structure ```per_epsilon_{dataset}_{paper}.csv``` contain the results for each image for each investigated epsilon on both training and testing set for each network investigated. 
- **experiments**: Folder containing the code for running ab-crown experiments for all datasets. Note that we currently do not include the code for running the BAB experiments as we are integrating this still in the VERONA package.
- **training**: Code for the training and PyTorch implementations for the networks
- **network_onnx**: Network in the onnx formats
  - [MNIST](networks_onnx/mnist)
    - [mnist_6_256_fgsm.onnx](networks_onnx/mnist/mnist_6_256_fgsm.onnx)
    - [mnist_6_256_pgd.onnx](networks_onnx/mnist/mnist_6_256_pgd.onnx)
    - [mnist_6_256_standard.onnx](networks_onnx/mnist/mnist_6_256_standard.onnx)
    - [mnist_7_1024_fgsm.onnx](networks_onnx/mnist/mnist_7_1024_fgsm.onnx)
    - [mnist_7_1024_pgd.onnx](networks_onnx/mnist/mnist_7_1024_pgd.onnx)
    - [mnist_7_1024_standard.onnx](networks_onnx/mnist/mnist_7_1024_standard.onnx)
    - [convSmallRELU__Point.onnx](networks_onnx/mnist/convSmallRELU__Point.onnx)
    - [convSmallRELU__DiffAI.onnx](networks_onnx/mnist/convSmallRELU__DiffAI.onnx)
    - [convSmallRELU__PGDK.onnx](networks_onnx/mnist/convSmallRELU__PGDK.onnx)
    - [convMedGRELU__Point.onnx](networks_onnx/mnist/convMedGRELU__Point.onnx)
    - [convMedGRELU__PGDK_w_0.1.onnx](networks_onnx/mnist/convMedGRELU__PGDK_w_0.1.onnx)
    - [convMedGRELU__PGDK_w_0.3.onnx](networks_onnx/mnist/convMedGRELU__PGDK_w_0.3.onnx)
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
    - [gtsrb_cnn_deep_game_standard.onnx](networks_onnx/gtsrb/gtsrb_cnn_deep_game_standard.onnx)
    - [gtsrb_cnn_deep_game_fgsm.onnx](networks_onnx/gtsrb/gtsrb_cnn_deep_game_fgsm.onnx)
    - [gtsrb_cnn_deep_game_pgd.onnx](networks_onnx/gtsrb/gtsrb_cnn_deep_game_pgd.onnx)
    - [gtsrb_cnn_vnncomp23_relu_standard.onnx](networks_onnx/gtsrb/gtsrb_cnn_vnncomp23_relu_standard.onnx)
    - [gtsrb_cnn_vnncomp23_relu_fgsm.onnx](networks_onnx/gtsrb/gtsrb_cnn_vnncomp23_relu_fgsm.onnx)
    - [gtsrb_cnn_vnncomp23_relu_pgd.onnx](networks_onnx/gtsrb/gtsrb_cnn_vnncomp23_relu_pgd.onnx)

## Reproduction instructions
Note that we have used the parallel execution from the VERONA package, but as every cluster and device is different we have added the sequential execution for each experiment in this repository. 

For each dataset we have added a main.py file in the corresponding [experiments](experiments/) folder. 



## External Packages
This project uses the following external packages:
- [VERONA](https://github.com/ADA-research/VERONA): An open-source package for creating Robustness Distributions.
- [adversarial-training-box](https://github.com/Aaron99B/adversarial-training-box): An open-source package for adversarial training of neural networks with PyTorch.



