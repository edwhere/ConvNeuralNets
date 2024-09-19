# Implementations of Convolutional Neural Networks

This code repository includes software for training Convolutional Neural Networks (CNN):

### train_resnet_ignite.py
Train a ResNet neural network for image classification. This implementation has the following characteristics:
- It uses the Pytorch Ignite framework, which facilitates development with functional high-level APIs: https://pytorch.org/ignite/index.html
- It runs on CPU, Nvidia GPU, or in the Apple GPU (found in M1, M2 and M3 machines)

## train_simplenet.py
Train a small and simple convolutional neural network. This implementation has the following characteristic:
- It runs on CPU, Nvidia GPU, or in the Apple GPU (found in M1, M2 and M3 machines)

## Command-line arguments
Enter `python train_resnet_ignite.py -h` or `python train_simplenet.py - h` in the command line for a description of required and optional arguments.

## License
This project is licensed under the MIT License (Expat version)

Copyright (c) 2024 - 2030 Edwin Heredia

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
