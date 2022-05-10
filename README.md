# Exploring Test Time Optimization for Discrete Neural Representation Learners

Our code can be split up into two implementation sections -- one on Normalising Flows, and one on the PyTorch VQVAE.

## Normalising Flows
We built this code on original code at https://github.com/VincentStimper/normalizing-flows with initial authors.

1. Vincent Stimper
2. Lukas Ryll
3. David Liu

Our contributions include
1. A Gumbel-Softmax target distribution class
2. Code for training the flows

## PyTorch VQ-VAE 

We built this code on original code at https://github.com/ritheshkumar95/pytorch-vqvae with
initial authors.

1. Rithesh Kumar
2. Tristan Deleu
3. Evan Racah

Our contributions include
1. Changed way of doing stop-gradient
2. Code for test-time optimizations
3. Visualizing/sampling code 
4. Extension of PixelCNN prior to other datasets.
5. Method for Normalizing Flows
