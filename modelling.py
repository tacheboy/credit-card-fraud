import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import load_data, preprocess_data, create_preprocessor

def feature_importance_analysis(pipeline, X_train, model_name):
    """
    Compute and visualize feature importance for trained models.
    Constructs the full list of feature names from the preprocessor.
    """
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessing']
    
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            try:
                new_names = list(transformer.get_feature_names_out(columns))
            except Exception:
                new_names = list(columns)
            feature_names.extend(new_names)
        else:
            feature_names.extend(list(columns))
    
    if model_name == "Random Forest":
        importance = classifier.feature_importances_
        sorted_idx = np.argsort(importance)[-10:]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title(f"Top Features - {model_name}")
        plt.show()

    elif model_name == "LogisticRegression":
        coef = classifier.coef_[0]
        sorted_idx = np.argsort(np.abs(coef))[-10:]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(sorted_idx)), coef[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Coefficient Weight")
        plt.title(f"Top Features - {model_name}")
        plt.show()


def shap_analysis(model, X_sample):
    """
    Perform SHAP analysis by transforming X_sample using the pipeline's preprocessor.
    This ensures that the explainer receives numeric data.
    """
    # Get the preprocessor from the pipeline
    preprocessor = model.named_steps['preprocessing']
    
    # Transform X_sample to get numeric feature values
    X_sample_trans = preprocessor.transform(X_sample)
    
    full_feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            try:
                new_names = list(transformer.get_feature_names_out(columns))
            except Exception:
                new_names = list(columns)
            full_feature_names.extend(new_names)
        else:
            full_feature_names.extend(list(columns))
    
    # Create SHAP explainer using the classifier and the transformed data
    explainer = shap.Explainer(model.named_steps['classifier'], X_sample_trans)
    shap_values = explainer(X_sample_trans)
    
    # Plot SHAP summary plot
    shap.summary_plot(shap_values, X_sample_trans, feature_names=full_feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.show()

def misclassification_analysis(y_test, y_pred, X_test):
    """
    Analyze misclassified instances (false positives&false negatives).
    """
    X_test = X_test.copy()
    X_test['actual'] = y_test
    X_test['predicted'] = y_pred

    false_positives = X_test[(X_test['actual'] == 0) & (X_test['predicted'] == 1)]
    false_negatives = X_test[(X_test['actual'] == 1) & (X_test['predicted'] == 0)]

    print(f"False Positives (Legitimate transactions misclassified as fraud): {len(false_positives)}")
    print(f"False Negatives (Fraudulent transactions misclassified as legitimate): {len(false_negatives)}")

    print("\nTop 5 False Positives:")
    print(false_positives.head())

    print("\nTop 5 False Negatives:")
    print(false_negatives.head())

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train models, evaluate performance, and feature importance.
    """
    preprocessor = create_preprocessor(X_train)

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

        # Train the model
        pipeline.fit(X_train, y_train)

        # Predict on test set
        preds = pipeline.predict(X_test)

        # Evaluate
        accuracy = np.mean(preds == y_test)
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:\n", classification_report(y_test, preds))

        # Feature Importance Analysis
        feature_importance_analysis(pipeline, X_train, name)

        # SHAP Analysis
        shap_analysis(pipeline, X_train.sample(500))  # Subset for SHAP due to compute cost

        # Misclassification Analysis
        misclassification_analysis(y_test, preds, X_test)

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = (name, pipeline)

    if best_model:
        print(f"\nBest Model: {best_model[0]} with accuracy {best_accuracy:.4f}")

def main():
    train_path = "./fraudTrain.csv"
    test_path = "./fraudTest.csv"
    train_df, test_df = load_data(train_path, test_path)

    X_train, y_train = preprocess_data(train_df)
    X_test, y_test = preprocess_data(test_df)

    train_and_evaluate(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
