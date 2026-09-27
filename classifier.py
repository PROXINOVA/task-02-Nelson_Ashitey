"""
Project 2: Data Classification Using AI
-----------------------------------------
A basic supervised-learning classification model built with a small,
well-known dataset (Iris flowers). Demonstrates:
  - Loading and understanding a dataset
  - Splitting data into training and testing sets
  - Applying a simple classification algorithm (Decision Tree)
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_and_explore_data():
    """Step 1: Load the dataset and print basic info to understand it."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = [iris.target_names[i] for i in iris.target]

    print("=" * 55)
    print(" STEP 1: Load and Understand the Dataset")
    print("=" * 55)
    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nClass distribution (species counts):")
    print(df["species"].value_counts())
    print("\nBasic statistics:")
    print(df.describe())

    return iris.data, iris.target, iris.target_names


def split_data(X, y):
    """Step 2: Split data into training and testing sets."""
    print("\n" + "=" * 55)
    print(" STEP 2: Split Data into Training and Testing Sets")
    print("=" * 55)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Step 3: Apply a simple classification algorithm."""
    print("\n" + "=" * 55)
    print(" STEP 3: Train the Classification Model")
    print("=" * 55)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("\nModel trained: Decision Tree Classifier")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")

    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Step 4: Evaluate the model on unseen test data."""
    print("\n" + "=" * 55)
    print(" STEP 4: Evaluate Model Performance")
    print("=" * 55)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy on test data: {accuracy * 100:.2f}%")
    print("\nDetailed classification report:")
    print(classification_report(y_test, predictions, target_names=target_names))


def predict_new_sample(model, target_names):
    """Bonus: show how the trained model classifies a brand-new sample."""
    print("=" * 55)
    print(" BONUS: Classify a New, Unseen Flower Sample")
    print("=" * 55)

    # Example measurements: [sepal length, sepal width, petal length, petal width]
    new_sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(new_sample)
    predicted_species = target_names[prediction[0]]

    print(f"\nNew sample measurements: {new_sample[0]}")
    print(f"Predicted species: {predicted_species}")


def main():
    X, y, target_names = load_and_explore_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, target_names)
    predict_new_sample(model, target_names)


if __name__ == "__main__":
    main()
