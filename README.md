# Image Classification with Deep Neural Network

## Overview

This repository contains the implementation of a deep neural network for image classification using the Fashion_mnist dataset from TensorFlow. The dataset consists of 10 classes, and the model is built as a Convolutional Neural Network (CNN).

## Model Architecture

The CNN model consists of:
- Two convolutional layers with 32 and 64 filters and ReLU activation.
- Two Max Pooling layers for dimensionality reduction.
- Two dense layers, one with 128 neurons and another with 10 neurons (output layer).
- Adam Optimizer is used for compilation.

### Training Details
- Epochs: 5
- Validation data: 10%
- Run Time: 2 minutes and 38 seconds
- Accuracy: 90.19%

## Convolutional Neural Network (CNN)

CNNs are designed for processing structured grid data, such as images. They have proven effective in various computer vision tasks, with image classification being one of their common applications.

## Hyperparameter Tuning

### Keras Tuner Library

The model's hyperparameters are tuned using the Keras Tuner Library, focusing on:
- Learning rate
- Number of units in the dense layer
- Epochs

The best hyperparameters are determined based on the highest validation accuracy.

### Bayesian Optimization

Bayesian Optimization is employed for hyperparameter tuning, utilizing a probabilistic model to guide the search efficiently.

- Run Time: 2 minutes and 41 seconds
- Accuracy: 89.59%
- Time for Trials: 12 minutes and 8 seconds
- Test Loss: 0.28

### GridSearch

GridSearch exhaustively evaluates all possible hyperparameter combinations within a predefined grid.

- Run Time: 2 minutes and 6 seconds
- Accuracy: 89.34%
- Time for Trials: 10 minutes and 31 seconds
- Test Loss: 0.3

### RandomSearch

RandomSearch randomly samples hyperparameter combinations, providing a more efficient alternative to grid search.

- Run Time: 4 minutes and 15 seconds
- Accuracy: 89.56%
- Time for Trials: 13 minutes and 29 seconds
- Test Loss: 0.28
