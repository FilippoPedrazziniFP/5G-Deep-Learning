# 5G-Deep-Learning
The repository contains the project did for the Research Methodologies course at @KTH (EIT Data Science Master Programme)

## Description
The aim of the project was to develop an intrusion detection system using Deep Learning Algorithms. The main implementation is done using Tensorflow following the feature work of another [paper](http://www.covert.io/research-papers/deep-learning-security/A%20Deep%20Learning%20Approach%20for%20Network%20Intrusion%20Detection%20System.pdf).

### The solution
The proposed solution is a Deep Neural Network composed by a Stacked Denoising Autoencoder in the first layers to extract the most meaningful features followed by a Simple Softmax Regression for classification purposes. 

TODO - adding the image of the network

### Further descriptions
The repository contains the presentation and the paper written for the project regarding the implementation and the results.

### Prerequisites
- Python 3.5
- Tensorflow 1.3
- Numpy
- Matplotlib
- Pandas
- Scikitlearn

### Usage 
For training and see the hold out results

	$ python3 main.py

For training and see the results on the testset

	$ python3 main.py --test=True





