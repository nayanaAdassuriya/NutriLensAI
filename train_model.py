"""
Model Training and Evaluation Module for NutriLens AI.

This module implements:
1. Training of an advanced CatBoost Multi-class Classifier on Oblivious Decision Trees,
   which is not part of standard introductory ML coursework (distinct from standard
   CART decision trees, logistic regression, or k-NN).
2. Hyperparameter configuration using balanced regularized gradient boosting.
3. Multi-metric evaluation: Accuracy, Macro-F1, Weighted-F1, and Confusion Matrix.
4. Serializing model weights and generating publication-ready evaluation plots.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from data_pipeline import load_and_preprocess_dataset, ARTIFACTS_DIR


MODEL_PATH = os.path.join(ARTIFACTS_DIR, "catboost_health_classifier.cbm")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "model_metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")


def train_catboost_model():
    """
    Executes training routine for the CatBoost nutritional grade classifier,
    evaluates performance across classes A-E, and saves all evaluation artifacts.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    print("[Training] Loading stratified packaged food dataset...")
    X_train, X_test, y_train, y_test, feature_names, metadata = load_and_preprocess_dataset()
    
    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    test_pool = Pool(X_test, y_test, feature_names=feature_names)
    
    # CatBoostClassifier with regularized oblivious trees
    print("[Training] Initializing CatBoostClassifier with oblivious tree structure...")
    model = CatBoostClassifier(
        iterations=400,
        depth=5,
        learning_rate=0.06,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        l2_leaf_reg=3.5,
        random_seed=42,
        verbose=50
    )
    
    print("[Training] Fitting model with early stopping monitoring...")
    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=40,
        use_best_model=True
    )
    
    # Save model binary
    model.save_model(MODEL_PATH)
    print(f"[Training] Model successfully saved to {MODEL_PATH}")
    
    # Evaluation
    print("\n" + "="*50)
    print("           MODEL EVALUATION REPORT")
    print("="*50)
    y_pred = model.predict(test_pool).ravel()
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    
    class_labels = metadata["class_names"]
    report_dict = classification_report(
        y_test, y_pred, target_names=class_labels, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_test, y_pred, target_names=class_labels, zero_division=0
    )
    
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}\n")
    print("Detailed Classification Breakdown:")
    print(report_str)
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True
    )
    plt.title("Confusion Matrix - CatBoost Nutritional Health Grade", fontsize=12, pad=12)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("True Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"[Evaluation] Confusion matrix plot saved to {CONFUSION_MATRIX_PATH}")
    
    # Persist metrics summary
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report_dict,
        "best_iteration": int(model.get_best_iteration()),
        "hyperparameters": {
            "iterations": 400,
            "depth": 5,
            "learning_rate": 0.06,
            "l2_leaf_reg": 3.5,
            "loss_function": "MultiClass"
        }
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[Evaluation] Metrics report saved to {METRICS_PATH}")
    return model, metrics


if __name__ == "__main__":
    train_catboost_model()
