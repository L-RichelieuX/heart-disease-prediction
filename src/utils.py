"""
Utility functions for model evaluation and visualization.

Provides functions for:
- Training and evaluating multiple classifiers
- Plotting confusion matrices and ROC curves
- Generating SHAP explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from sklearn.model_selection import GridSearchCV, cross_val_score


# ═══════════════════════════════════════════════════════════════════════
# Global style settings
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]


# ═══════════════════════════════════════════════════════════════════════
# Model training & evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a trained classifier on test data.

    Parameters
    ----------
    model : estimator
        Trained scikit-learn classifier.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True test labels.

    Returns
    -------
    dict
        Dictionary with accuracy, precision, recall, f1, and auc.
    """
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_test, y_proba)

    return metrics


def evaluate_all_models(
    models: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Train and evaluate multiple models, returning a summary DataFrame.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to (unfitted) estimator instances.
    X_train, X_test, y_train, y_test : np.ndarray
        Train/test data.

    Returns
    -------
    pd.DataFrame
        Summary table of all metrics across models.
    """
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)

    df_results = pd.DataFrame(results)
    df_results = df_results.set_index("model")
    return df_results


def cross_validate_all_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """
    Perform cross-validation (no test set) for multiple models.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to (unfitted) estimator instances.
    X_train, y_train : np.ndarray
        Training data (full, before test split).
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric (e.g., 'roc_auc', 'accuracy').

    Returns
    -------
    pd.DataFrame
        Mean and std of CV scores for each model.
    """
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        results.append({
            "Model": name,
            f"Mean {scoring.upper()}": round(scores.mean(), 4),
            f"Std {scoring.upper()}": round(scores.std(), 4),
        })
    df_cv = pd.DataFrame(results).set_index("Model")
    return df_cv


def hyperparameter_tuning(
    model,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> tuple:
    """
    Perform grid-search hyperparameter tuning.

    Parameters
    ----------
    model : estimator
        Base model.
    param_grid : dict
        Parameter grid for GridSearchCV.
    X_train, y_train : np.ndarray
        Training data.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric.

    Returns
    -------
    tuple
        (best_estimator, best_params, best_score, cv_results_dataframe)
    """
    grid = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_, grid.cv_results_


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_path: str = None,
):
    """
    Plot a confusion matrix for a trained model.

    Parameters
    ----------
    model : estimator
        Trained classifier.
    X_test, y_test : np.ndarray
        Test data.
    model_name : str
        Display name for the plot title.
    save_path : str, optional
        Path to save the figure.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_roc_curves(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = None,
):
    """
    Plot ROC curves for multiple models on the same figure.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained estimators.
    X_test, y_test : np.ndarray
        Test data.
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, model) in enumerate(models.items()):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=COLORS[i % len(COLORS)])

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_metrics_bar(df_results: pd.DataFrame, save_path: str = None):
    """
    Plot a grouped bar chart comparing model metrics.

    Parameters
    ----------
    df_results : pd.DataFrame
        Results DataFrame from evaluate_all_models().
    save_path : str, optional
        Path to save the figure.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    if "auc" in df_results.columns:
        metrics.append("auc")

    df_plot = df_results[metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black")
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 15,
    model_name: str = "Model",
    save_path: str = None,
):
    """
    Plot feature importance for tree-based models.

    Parameters
    ----------
    model : estimator
        Trained tree-based classifier with feature_importances_ attribute.
    feature_names : list
        List of feature names.
    top_n : int
        Number of top features to show.
    model_name : str
        Display name.
    save_path : str, optional
        Path to save the figure.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"Model {model_name} does not have feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color="#3498db",
        edgecolor="black",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
