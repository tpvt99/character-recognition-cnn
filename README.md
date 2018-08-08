# Introduction

This project is using convolution neural network to recognize chracter from 0-9, a-z and A-Z  
Python 3+ and Tensorflow is required for training and testing

## Getting Started

These instructions will help you run this project on a Linux machine.  

### Prequisites

Data of this project is downloaded from https://www.nist.gov/srd/nist-special-database-19.  

If you don't want to download database on this website, you should use the training data I provide in folder data_base/

### Steps
Running load_data.py to generate all the data requires to recognize number from 0 to 9

Then run cnn.py to train

Last, run test.py to recognize the character. You need a camera, a paper with a number you draw on it. Move the paper to the near webcam and see the result.
