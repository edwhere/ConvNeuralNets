# Implementations of Convolutional Neural Networks

This code repository includes software for training Convolutional Neural Networks (CNN):

### train_resnet_ignite.py
Train a ResNet neural network for image classification. This implementation has the following characteristics:
- It uses the Pytorch Ignite framework, which facilitates development with functional high-level APIs: https://pytorch.org/ignite/index.html
- It runs on CPU, Nvidia GPU, or in the Apple GPU (found in M1, M2 and M3 machines)

## train_simplenet.py
Train a small and simple convolutional neural network. This implementation has the following characteristic:
- It runs on CPU, Nvidia GPU, or in the Apple GPU (found in M1, M2 and M3 machines)

## Function arguments
Enter `python train_resnet_ignite.py -h` or `python train_simplenet.py - h` in the command line for a description of required and optional arguments.
