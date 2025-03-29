# modeling.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import load_data, preprocess_data, create_preprocessor
from visualization import basic_eda, plot_fraud_distribution

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot a confusion matrix.
    """
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train models on training data and evaluate them on testing data.
    """

    # Preprocessing Pipeline
    preprocessor = create_preprocessor(X_train)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }

    best_model = None
    best_accuracy = 0.0

    for name, model in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])

        # Train on training set
        pipeline.fit(X_train, y_train)

        # Predict on test set
        preds = pipeline.predict(X_test)

        # Evaluate
        accuracy = np.mean(preds == y_test)
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:\n", classification_report(y_test, preds))

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        plot_confusion_matrix(cm, classes=[0,1], title=f"{name} Confusion Matrix")

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = (name, pipeline)

    # Final best model report
    if best_model:
        print(f"\nBest Model: {best_model[0]} with accuracy {best_accuracy:.4f}")

def main():
    """
    Main function to load, preprocess, and train models.
    """
    # Load data
    train_path = "./fraudTrain.csv"  # Adjust path as needed
    test_path = "./fraudTest.csv"  # Adjust path as needed
    train_df, test_df = load_data(train_path, test_path)

    # EDA
    print("\nTraining Data:")
    basic_eda(train_df)
    plot_fraud_distribution(train_df)

    print("\nTesting Data:")
    basic_eda(test_df)
    plot_fraud_distribution(test_df)

    # Preprocess data
    X_train, y_train = preprocess_data(train_df)
    X_test, y_test = preprocess_data(test_df)

    # Train & Evaluate Models
    train_and_evaluate(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
