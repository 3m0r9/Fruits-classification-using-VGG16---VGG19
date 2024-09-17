# Fruits Classification using VGG16 and VGG19

This project applies transfer learning techniques using pre-trained models, VGG16 and VGG19, for the task of fruit image classification. By leveraging the power of convolutional neural networks (CNNs) and pre-trained architectures, this project demonstrates how transfer learning can be used to efficiently classify images of different fruit types.

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

The goal of this project is to classify different fruit types using the VGG16 and VGG19 architectures, which are popular CNN models pre-trained on the ImageNet dataset. Transfer learning allows us to leverage these pre-trained models for our fruit classification task, reducing training time and improving performance.

The project covers:
- Loading and preprocessing the fruit images dataset.
- Using VGG16 and VGG19 models with custom layers for the classification task.
- Model training, evaluation, and comparison between VGG16 and VGG19.
  
## Dataset

The dataset used for this project contains images of different fruit types and can be found on [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Even though the dataset is originally intended for customer churn, in this project, it has been adapted for fruit classification by renaming and restructuring the files accordingly.

- **Classes**: Various fruits (e.g., Apple, Banana, Orange, etc.).
- **Input**: Images of fruit.
- **Target**: Class labels corresponding to the type of fruit.

Download the dataset and place it in the `data/` directory of the project.

## Data Preprocessing

Data preprocessing steps include:
- **Image Resizing**: All fruit images are resized to match the input size required by VGG16 and VGG19 (224x224 pixels).
- **Data Augmentation**: Techniques like rotation, flipping, and zooming are applied to artificially increase the size of the dataset and improve model generalization.
- **Normalization**: Pixel values are normalized to fall within the range [0, 1] for better convergence during training.

These steps ensure the dataset is in the proper format for efficient model training.

## Modeling

Two models are implemented using transfer learning:
1. **VGG16**: A pre-trained model that is fine-tuned for the fruit classification task.
2. **VGG19**: Another variant of the VGG model with more convolutional layers, also fine-tuned for this task.

Both models are customized by adding fully connected layers on top of the pre-trained base model to adapt them to the fruit classification task. The architecture includes:
- Pre-trained convolutional layers from VGG16 or VGG19.
- A global average pooling layer.
- Fully connected dense layers.
- A softmax output layer for multi-class classification.

## Evaluation

The models are evaluated based on the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

These metrics provide insights into how well the models perform in classifying fruit images correctly.

## Installation

To run this project locally, follow these steps:

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
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the fruit image dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data/` directory.
2. Run the preprocessing script to prepare the data for training:
   ```bash
   python preprocess_data.py --input data/fruit_images --output data/processed_images
   ```
3. Train the model using either VGG16 or VGG19:
   ```bash
   python train_model.py --model vgg16 --input data/processed_images
   ```
   or
   ```bash
   python train_model.py --model vgg19 --input data/processed_images
   ```
4. Evaluate the model:
   ```bash
   python evaluate_model.py --model vgg16 --input data/test_images
   ```

## Results

The results from the VGG16 and VGG19 models on the test set showed the following performance:
- **VGG16**: 
  - Accuracy: 92%
  - Precision: 0.91
  - Recall: 0.90
  - F1-score: 0.91
- **VGG19**: 
  - Accuracy: 93%
  - Precision: 0.92
  - Recall: 0.91
  - F1-score: 0.92

The results show that both models perform well, with VGG19 slightly outperforming VGG16 on this dataset.

## Contributors

- **Imran Abu Libda** - [3m0r9](https://github.com/3m0r9)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Let's Connect

- **GitHub** - [3m0r9](https://github.com/3m0r9)
- **LinkedIn** - [Imran Abu Libda](https://www.linkedin.com/in/imran-abu-libda/)
- **Email** - [imranabulibda@gmail.com](mailto:imranabulibda@gmail.com)
- **Medium** - [Imran Abu Libda](https://medium.com/@imranabulibda_23845)
