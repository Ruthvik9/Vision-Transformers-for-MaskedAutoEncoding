# Vision-Transformers-for-MaskedAutoEncoding


![image](https://github.com/Ruthvik9/Vision-Transformers-for-MaskedAutoEncoding/assets/74010232/23f3c3e1-a35d-4192-b01d-da0187154f67)


This repository contains my implementation of the Masked AutoEncoder (MAE) for self-supervised learning, as proposed in the paper "Masked Autoencoders Are Scalable Vision Learners" by , Facebook AI Research (FAIR.) The model is implemented in PyTorch.

## Introduction
The Masked AutoEncoder (MAE) is a self-supervised learning model that learns to reconstruct masked portions of an image. The model is based on the Transformer architecture and uses a masked autoencoder setup similar to BERT. The model is trained to predict the pixel values of masked patches, given the remaining unmasked patches of the image.

## Model Architecture
The MAE consists of two main components: an encoder and a decoder. Both the encoder and decoder are based on the Transformer architecture. The encoder takes in the unmasked patches of an image and generates high-dimensional embeddings. These embeddings, along with embeddings of the masked patches (referred to as "mask tokens"), are then passed through the decoder to reconstruct the original image.
The encoder and decoder are made up of multiple blocks, each containing a multi-head self-attention mechanism and a feed-forward neural network. The blocks also include layer normalization and dropout for regularization.


![image](https://github.com/Ruthvik9/Vision-Transformers-for-MaskedAutoEncoding/assets/74010232/28b6bc84-2ae2-4065-97a3-1fe6c8c1e441)


## Training
The model was trained using the Mean Squared Error (MSE) loss between the reconstructed and original images. The loss was computed **only on the masked patches**, similar to BERT.

## Results and Discussion
The model was trained on a dataset of images, and the loss was observed to decrease over time, indicating that the model was learning to reconstruct the masked patches. However, the loss plateaued after a certain number of epochs, suggesting that the model had reached its capacity to learn from the given dataset.
