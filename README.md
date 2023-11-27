
# Autonomous Vehicle Steering Prediction

This repository contains code for training a neural network model to predict steering angles for autonomous vehicles. The model is trained using the NVIDIA End-to-End Self-Driving Car architecture on a dataset obtained from a simulated environment.

The code is written in Python and utilizes the Keras library for building and training the neural network. The training data includes images captured from the vehicle's cameras and corresponding steering angles.

## Code Overview

The main code is provided in the Jupyter Notebook file `final_code.ipynb`. The notebook is structured as follows:

1. **Data Preparation**: Loading and preprocessing the training data, including balancing the dataset for steering angles.
2. **Data Augmentation**: Implementing image augmentation techniques such as zooming, panning, brightness alteration, and flipping to enhance the dataset.
3. **Image Preprocessing**: Defining a function for image preprocessing, including cropping, color space conversion, Gaussian blurring, resizing, and normalization.
4. **Model Architecture**: Implementing the NVIDIA End-to-End Self-Driving Car architecture for steering angle prediction.
5. **Training**: Training the model using a batch generator to handle augmented data on-the-fly.
6. **Evaluation**: Visualizing and analyzing the training and validation loss over epochs.
7. **Saving the Model**: Saving the trained model as `model.h5`.

## Dependencies

Ensure you have the required dependencies installed before running the code:


