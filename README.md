# End-to-End-AI-Communication-System
Compared the traditional QPSK modulation technique with the new AI Autoencoder

Project Overview
This project compares a traditional digital communication technique, QPSK (Quadrature Phase Shift Keying), with a deep learning-based Autoencoder Neural Network over an AWGN (Additive White Gaussian Noise) channel.
The objective of this project is to analyze how AI-based communication systems can improve signal transmission performance compared to conventional modulation schemes under noisy channel conditions.

Features
Simulation of QPSK communication system
Deep learning-based autoencoder communication model
AWGN channel implementation
BER (Bit Error Rate) performance comparison
SNR vs BER graphical analysis
TensorFlow-based neural network training

What is an Autoencoder?
An autoencoder is a type of neural network that learns how to compress input data into a smaller representation and then reconstruct it back with minimum loss.
It consists of:
Encoder – Compresses the input data
Channel Layer – Adds AWGN noise
Decoder – Reconstructs the original message

System Workflow
Input Bits
    ↓
Encoder Network
    ↓
Encoded Signal
    ↓
AWGN Channel
    ↓
Decoder Network
    ↓
Recovered Bits
    ↓
BER Calculation & Performance Analysisa

Programming Language
Python

Libraries Used

1. TensorFlow
Used for:
  Building the autoencoder neural network
  Training and optimization
  Backpropagation

2. NumPy
Used for:
  Numerical computations
  Random bit generation
  Matrix operations

3. Matplotlib
Used for:
  Plotting BER vs SNR graphs
  isualization of performance comparison
