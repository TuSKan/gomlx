# AlexNet Model

![alt text](./alexnet2.png "AlexNet Archtecture")

## Modifications

This library creates the AlexNet model with the folling modification:
    1) The original was architected to operate distributedly across two systems (this implementation is not distributed)
    2) Batch Normalization is used in-place of the original Alexnet's local response normalization

## Preprocessing

In terms of preprocessing, the images were resized, cropped, and normalized. To normalize the images, the mean and start deviation of the pixels RGB data is used as provided by Krizhevsky et al (mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]).

## Reference

[ImageNet Classification with Deep Convolutional Neural Networks](
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (NIPS 2012)
