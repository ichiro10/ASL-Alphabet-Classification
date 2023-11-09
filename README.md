# American Sign Language Alphabet Classification

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Getting Started](#getting-started)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
This repository contains code for training and evaluating a deep learning model for classifying alphabets from the American Sign Language (ASL). The model uses transfer learning on a pre-trained neural network to achieve high accuracy in ASL alphabet recognition.

## Dataset
The dataset used in this project is a collection of ASL alphabet images. Here are some details about the dataset:

- **Training Data:** Contains 87,000 images, each of size 200x200 pixels. The dataset is divided into 29 folders, representing 29 classes.
- **Classes:** There are 29 classes in total. This includes 26 classes for the letters A-Z and 3 additional classes for SPACE, DELETE, and NOTHING. These additional classes are crucial for real-time applications and classification.
- [Link to Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/da)
## Model
The ASL alphabet classification model in this project is built using transfer learning. We fine-tune a pre-trained neural network on the ASL alphabet dataset to achieve high accuracy. The model architecture and weights can be found in the code.

## Getting Started
To get started with this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/ASL-Alphabet-Classification.git
   cd ASL-Alphabet-Classification
