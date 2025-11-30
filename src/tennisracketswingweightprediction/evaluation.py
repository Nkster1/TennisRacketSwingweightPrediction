"""Evaluation utilities for tennis racket swingweight prediction models."""

from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.pipeline import make_pipeline


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics for predictions.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Dictionary containing various regression metrics
    """
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    med_ae = metrics.median_absolute_error(y_true, y_pred)

    # R² score
    r2 = metrics.r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error (if no zeros in y_true)
    mape = None
    if np.all(y_true != 0):
        mape = metrics.mean_absolute_percentage_error(y_true, y_pred)

    # Max error
    max_error = metrics.max_error(y_true, y_pred)

    # Explained variance score
    explained_var = metrics.explained_variance_score(y_true, y_pred)

    # Mean Squared Log Error (if all values are non-negative)
    msle = None
    if np.all(y_true >= 0) and np.all(y_pred >= 0):
        msle = metrics.mean_squared_log_error(y_true, y_pred)

    # Residual statistics
    residuals = y_true - y_pred
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    residual_median = np.median(residuals)
    residual_mad = calculate_mad(residuals)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "median_ae": med_ae,
        "r2": r2,
        "mape": mape,
        "max_error": max_error,
        "explained_variance": explained_var,
        "msle": msle,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "residual_median": residual_median,
        "residual_mad": residual_mad,
    }


def calculate_mad(x: np.ndarray) -> float:
    """
    Calculate Median Absolute Deviation (MAD) from median.

    Args:
        x: Input array

    Returns:
        Median absolute deviation
    """
    return np.median(np.absolute(x - np.median(x)))


def plot_diagnostics(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Create comprehensive diagnostic plots for regression analysis.

    Args:
        y_train: True training target values
        y_train_pred: Predicted training target values
        y_test: True test target values
        y_test_pred: Predicted test target values
        figsize: Figure size
    """
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1. Residual Distribution - Train
    axes[0, 0].hist(
        train_residuals, bins=50, edgecolor="black", alpha=0.7, color="blue"
    )
    axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0, 0].axvline(
        np.median(train_residuals),
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Median",
    )
    axes[0, 0].set_xlabel("Residuals")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Train Set: Residual Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residual Distribution - Test
    axes[0, 1].hist(
        test_residuals, bins=50, edgecolor="black", alpha=0.7, color="green"
    )
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0, 1].axvline(
        np.median(test_residuals),
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Median",
    )
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Test Set: Residual Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Q-Q Plot - Test (to check normality of residuals)
    from scipy import stats

    stats.probplot(test_residuals, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title("Test Set: Q-Q Plot")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Predicted vs Actual - Train
    axes[1, 0].scatter(y_train, y_train_pred, alpha=0.5, s=20, color="blue")
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    axes[1, 0].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    axes[1, 0].set_xlabel("True Values")
    axes[1, 0].set_ylabel("Predicted Values")
    axes[1, 0].set_title("Train Set: Predicted vs Actual")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Predicted vs Actual - Test
    axes[1, 1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color="green")
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    axes[1, 1].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    axes[1, 1].set_xlabel("True Values")
    axes[1, 1].set_ylabel("Predicted Values")
    axes[1, 1].set_title("Test Set: Predicted vs Actual")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Residuals vs Predicted - Test (to check homoscedasticity)
    axes[1, 2].scatter(y_test_pred, test_residuals, alpha=0.5, s=20, color="green")
    axes[1, 2].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1, 2].set_xlabel("Predicted Values")
    axes[1, 2].set_ylabel("Residuals")
    axes[1, 2].set_title("Test Set: Residuals vs Predicted")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_model(
    model: RegressorMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    plot_diagnostics_flag: bool = True,
) -> Tuple[RegressorMixin, Dict[str, Any]]:
    """
    Evaluate a regression model on train and test sets.

    Args:
        model: Sklearn regressor to evaluate
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        plot_diagnostics_flag: Whether to plot diagnostic charts

    Returns:
        Tuple of (fitted_model, metrics_dict)
    """
    # Fit the model
    model = model.fit(X_train, y_train)

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics for both sets
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    # Calculate outliers for test set (3 MAD rule)
    test_residuals = y_test - y_test_pred
    test_residual_median = test_metrics["residual_median"]
    test_residual_mad = test_metrics["residual_mad"]
    lower_outliers = test_residuals < test_residual_median - 3 * test_residual_mad
    upper_outliers = test_residuals > test_residual_median + 3 * test_residual_mad
    n_outliers = np.sum(lower_outliers | upper_outliers)
    outlier_percentage = 100 * n_outliers / len(test_residuals)

    # Plot diagnostics if requested
    if plot_diagnostics_flag:
        plot_diagnostics(y_train, y_train_pred, y_test, y_test_pred)

    # Compile all metrics
    results = {
        "train": train_metrics,
        "test": test_metrics,
        "test_n_outliers": n_outliers,
        "test_outlier_percentage": outlier_percentage,
    }

    return model, results


def print_evaluation_results(results: Dict[str, Any]) -> None:
    """
    Pretty print evaluation results.

    Args:
        results: Dictionary of evaluation metrics from evaluate_model
    """
    print("=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)

    # Train metrics
    train = results["train"]
    print("\nTRAIN SET METRICS:")
    print(f"  R²:                 {train['r2']:.4f}")
    print(f"  MAE:                {train['mae']:.4f}")
    print(f"  RMSE:               {train['rmse']:.4f}")
    print(f"  Median AE:          {train['median_ae']:.4f}")
    print(f"  Max Error:          {train['max_error']:.4f}")
    print(f"  Explained Variance: {train['explained_variance']:.4f}")
    if train["mape"] is not None:
        print(f"  MAPE:               {train['mape']:.4f}")
    if train["msle"] is not None:
        print(f"  MSLE:               {train['msle']:.4f}")

    print("\n  Residual Statistics:")
    print(f"    Mean:             {train['residual_mean']:.4f}")
    print(f"    Std Dev:          {train['residual_std']:.4f}")
    print(f"    Median:           {train['residual_median']:.4f}")
    print(f"    MAD:              {train['residual_mad']:.4f}")

    # Test metrics
    test = results["test"]
    print("\nTEST SET METRICS:")
    print(f"  R²:                 {test['r2']:.4f}")
    print(f"  MAE:                {test['mae']:.4f}")
    print(f"  RMSE:               {test['rmse']:.4f}")
    print(f"  Median AE:          {test['median_ae']:.4f}")
    print(f"  Max Error:          {test['max_error']:.4f}")
    print(f"  Explained Variance: {test['explained_variance']:.4f}")
    if test["mape"] is not None:
        print(f"  MAPE:               {test['mape']:.4f}")
    if test["msle"] is not None:
        print(f"  MSLE:               {test['msle']:.4f}")

    print("\n  Residual Statistics:")
    print(f"    Mean:             {test['residual_mean']:.4f}")
    print(f"    Std Dev:          {test['residual_std']:.4f}")
    print(f"    Median:           {test['residual_median']:.4f}")
    print(f"    MAD:              {test['residual_mad']:.4f}")

    print("\n  Outlier Analysis (3 MAD rule):")
    print(f"    N Outliers:       {results['test_n_outliers']}")
    print(f"    Outlier %:        {results['test_outlier_percentage']:.2f}%")

    print("=" * 70)


def evaluate_multiple_models(
    models: List[RegressorMixin],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    plot_diagnostics_flag: bool = False,
    feature_scaler: Optional[TransformerMixin] = None,
    target_scaler: Optional[TransformerMixin] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate multiple regression models and return comparison dataframes.

    Args:
        models: List of sklearn regressors to evaluate
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        plot_diagnostics_flag: Whether to plot diagnostic charts
        feature_scaler: Optional scaler for features (e.g., StandardScaler())
        target_scaler: Optional scaler for target (e.g., StandardScaler())

    Returns:
        Tuple of (results_all, results_test) DataFrames
        - results_all: DataFrame with train and test metrics for all models
        - results_test: DataFrame with only test metrics, sorted by MAE
    """
    all_metrics = {}
    all_metrics_test = {}

    # Apply target scaling if specified
    if target_scaler is not None:
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_train_scaled = y_train
        y_test_scaled = y_test

    for model in models:
        # Wrap model with feature scaler if specified
        if feature_scaler is not None:
            model_to_evaluate = make_pipeline(feature_scaler, model)
        else:
            model_to_evaluate = model

        # Evaluate the model
        trained_model, metrics = evaluate_model(
            model_to_evaluate,
            X_train,
            y_train_scaled,
            X_test,
            y_test_scaled,
            plot_diagnostics_flag=plot_diagnostics_flag,
        )

        # Extract model name from string representation
        model_identifier = str(model).split("(")[0]

        # Store metrics
        all_metrics[f"{model_identifier}_train"] = metrics["train"]
        all_metrics[f"{model_identifier}_test"] = metrics["test"]
        all_metrics_test[f"{model_identifier}_test"] = metrics["test"]

    # Convert to DataFrames
    results_all = pd.DataFrame(all_metrics).T
    results_test = pd.DataFrame(all_metrics_test).T

    # Sort test results by MAE
    results_test = results_test.sort_values("mae")

    return results_all, results_test
