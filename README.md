# task-02-Nelson_Ashitey
This GitHub repository contains my second project assignment during my internship period at DecodeLabs
[README.md](https://github.com/user-attachments/files/32691988/README.md)
# Data Classification Using AI

A simple supervised-learning project that builds, trains, and evaluates a
classification model using scikit-learn's built-in Iris flower dataset.

## Overview

This project demonstrates the core workflow of a machine learning
classification task:

1. **Load and understand a dataset** – load the Iris dataset and explore
   its shape, sample rows, class distribution, and summary statistics.
2. **Split the data** – divide the dataset into training and testing sets
   (80% / 20%) using a stratified split.
3. **Train a model** – fit a Decision Tree classifier on the training data.
4. **Evaluate the model** – measure accuracy and generate a detailed
   classification report on unseen test data.
5. **Bonus: predict a new sample** – classify a brand-new, made-up flower
   measurement to show the trained model in action.

## Dataset

The project uses the **Iris dataset**, a classic beginner-friendly dataset
built into scikit-learn (no download required). It contains 150 samples of
iris flowers from three species — *setosa*, *versicolor*, and *virginica*
— each described by four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Algorithm

The model uses a **Decision Tree Classifier**, a simple and interpretable
algorithm that splits data based on feature thresholds to classify samples
into categories.

## Requirements

- Python 3.7+
- pandas
- scikit-learn

Install dependencies with:

```bash
pip install pandas scikit-learn
```

## How to Run

```bash
python classifier.py
```

The script will print output for each step directly to the console:
dataset exploration, train/test split sizes, model training details,
evaluation metrics, and a sample prediction.

## Example Output

```
STEP 1: Load and Understand the Dataset
Dataset shape: 150 rows, 5 columns
...

STEP 2: Split Data into Training and Testing Sets
Training samples: 120
Testing samples:  30

STEP 3: Train the Classification Model
Model trained: Decision Tree Classifier
Tree depth: 5
Number of leaves: 8

STEP 4: Evaluate Model Performance
Accuracy on test data: 93.33%
...

BONUS: Classify a New, Unseen Flower Sample
New sample measurements: [5.1, 3.5, 1.4, 0.2]
Predicted species: setosa
```

## Key Skills Demonstrated

- Data handling and exploration with pandas
- Supervised learning fundamentals
- Train/test splitting for model evaluation
- Training and evaluating a classification algorithm
- Making predictions on new, unseen data

## Project Structure

```
.
├── classifier.py   # Main script: loads data, trains, and evaluates the model
└── README.md       # Project documentation
```

## Possible Extensions

- Swap the Decision Tree for another algorithm (e.g. Random Forest, KNN,
  Logistic Regression) and compare results.
- Visualise the decision tree structure.
- Try the workflow on a different dataset (e.g. wine or breast cancer
  datasets, also built into scikit-learn).
