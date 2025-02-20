# BBM406 - Hacettepe University Computer Science Assignments

This repository contains four different assignments for the BBM406 course at Hacettepe University. Each assignment involves implementing and experimenting with various machine learning and deep learning techniques.

## Assignments Overview

### 1. Perceptron & Fisher Algorithm

- Implements **Perceptron** and **Fisher's Linear Discriminant** for classification tasks.
- Uses a dataset with numerical features for binary classification.
- Compares the performance of both algorithms on given datasets.

### 2. SVM & Logistic Regression

- Implements **Support Vector Machine (SVM)** and **Logistic Regression**.
- Uses a dataset related to **banknote authentication**, where the goal is to determine whether a banknote is real or counterfeit based on extracted features.
- Evaluates and compares their classification performance.

### 3. CNN with ResNet & MobileNet Fine-Tuning

- Uses **Convolutional Neural Networks (CNNs)** for image classification.
- Uses an image dataset, fine-tuning **ResNet** and **MobileNet** models to improve classification performance.

### 4. LSTM for Sentiment Analysis

- Implements a **Long Short-Term Memory (LSTM)** model for sentiment analysis.
- Uses a **text dataset** containing user reviews and aims to classify sentiment as positive or negative.
- Trains the model on a text dataset to classify sentiment.

## Installation & Dependencies

Ensure you have the required dependencies installed before running any of the assignments.

```bash
pip install -r requirements.txt
```

## Running the Assignments

Each assignment is contained in its respective directory. Navigate to the desired assignment folder and run the corresponding script.

Example:

```bash
cd assignment_1  # Change to the relevant folder
python perceptron.py  # Run the script
```

For deep learning models (CNN, LSTM), ensure that you have a GPU enabled environment for faster training.

## Repository Structure

```
BBM406-HU-CS/
│-- assignment_1/   # Perceptron & Fisher
│-- assignment_2/   # SVM & Logistic Regression
│-- assignment_3/   # CNN with ResNet & MobileNet
│-- assignment_4/   # LSTM for Sentiment Analysis
│-- requirements.txt
│-- README.md
```

## Author

[Kagan Soysal](https://github.com/kagansoysal)

## License

This project is for educational purposes. Feel free to use and modify it for learning purposes.

