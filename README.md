# Fruits Classification using VGG16 and VGG19

This project implements deep learning models, specifically VGG16 and VGG19, for classifying different types of fruits from images. The project demonstrates how transfer learning can be leveraged to classify fruit images with high accuracy.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributors](#contributors)
10. [License](#license)
11. [Let's Connect](#lets-connect)

## Project Overview

This project focuses on using pre-trained models (VGG16 and VGG19) for classifying fruit images. VGG16 and VGG19 are Convolutional Neural Network (CNN) architectures that are widely used for image classification tasks due to their performance and generalization capabilities.

The project showcases the following:
- Utilization of transfer learning via VGG16 and VGG19.
- Fine-tuning the models for fruit classification.
- Comparing the performance of VGG16 and VGG19 on the fruit dataset.

## Dataset

The dataset used in this project is the [Fruits 360 Dataset](https://www.kaggle.com/datasets/moltean/fruits), available on Kaggle. It contains over 70,000 images of fruits with 131 distinct categories.

- **Images**: 32x32 RGB images of fruits.
- **Classes**: 131 unique fruit categories.
- **Size**: 67,692 training images and 22,256 test images.

The dataset can be downloaded from Kaggle, and it should be placed in the `data/` directory for training and testing.

## Data Preprocessing

Before training the models, the images go through the following preprocessing steps:
- **Resizing**: Images are resized to match the input shape required by VGG16 and VGG19 (224x224 pixels).
- **Normalization**: Pixel values are normalized to scale them between 0 and 1.
- **One-Hot Encoding**: The labels are one-hot encoded for classification.
- **Data Augmentation**: Techniques like rotation, flipping, and zooming are applied to augment the dataset and reduce overfitting.

## Modeling

### VGG16 and VGG19

Both VGG16 and VGG19 are deep CNN architectures that have been pre-trained on the ImageNet dataset. In this project:
- We used transfer learning by leveraging the pre-trained weights of VGG16 and VGG19.
- The final fully connected layers were replaced with custom layers to adapt the network for fruit classification.
- Fine-tuning was applied to adjust the deeper layers for improved performance on this specific dataset.

### Model Structure:
- **Convolutional Layers**: Extract spatial features from the images.
- **Max-Pooling Layers**: Reduce the spatial dimensions while retaining important features.
- **Fully Connected Layers**: Map the features to the output categories.
- **Softmax Layer**: Produces a probability distribution for classifying the input image into one of the 131 fruit categories.

### Libraries Used:
- TensorFlow / Keras
- NumPy
- OpenCV (for image handling)
- Matplotlib (for plotting)

## Evaluation

The performance of both models was evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Additionally, the models were evaluated on a test set, and confusion matrices were plotted to visualize the classification results.

## Installation

To run this project on your local machine, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/3m0r9/Fruits-classification-using-VGG16---VGG19.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Fruits-classification-using-VGG16---VGG19
   ```
3. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/moltean/fruits) and placed in the `data/` directory.
2. Preprocess the dataset:
   ```bash
   python preprocess_data.py --input data/
   ```
3. Train the models:
   ```bash
   python train_model.py --model vgg16
   python train_model.py --model vgg19
   ```
4. Evaluate the models on the test set:
   ```bash
   python evaluate_model.py --model vgg16
   python evaluate_model.py --model vgg19
   ```

## Results

Both VGG16 and VGG19 models were able to classify the fruits with high accuracy. Below are the key results:

- **VGG16**: 91% accuracy on the test set.
- **VGG19**: 92% accuracy on the test set.

Further details on the performance metrics, confusion matrices, and training plots can be found in the `results/` directory.

## Contributors

- **Imran Abu Libda** - [3m0r9](https://github.com/3m0r9)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Let's Connect

- **GitHub** - [3m0r9](https://github.com/3m0r9)
- **LinkedIn** - [Imran Abu Libda](https://www.linkedin.com/in/imran-abu-libda/)
- **Email** - [imranabulibda@gmail.com](mailto:imranabulibda@gmail.com)
- **Medium** - [Imran Abu Libda](https://medium.com/@imranabulibda_23845)
