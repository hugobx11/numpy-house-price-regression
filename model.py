"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    X = np.asarray(X, dtype=float)
    N, F = X.shape
    columns_means = np.nanmean(X, axis=0) # shape (F,)
    columns_means = np.where(np.isnan(columns_means), 0.0, columns_means)
    return np.where(np.isnan(X), columns_means, X)

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    X = np.asarray(X, dtype=float)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X, a_min=lower, a_max=upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    col = np.reshape(col, (X.shape[0], -1))
    return np.concatenate([X, col], axis=1)

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)
    uniques = np.unique(labels)
    return (labels[:, None] == uniques[None, :]).astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std==0, 1.0, std)
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    N = X.shape[0]
    return np.concatenate([np.ones((N, 1)), X], axis=1)

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng = np.random.RandomState(seed)
    return rng.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    N = indices.shape[0]
    first_floor = int((N * train_ratio))
    second_floor = int((N * (train_ratio + val_ratio)))
    return (
        indices[:first_floor],
        indices[first_floor: second_floor],
        indices[second_floor:]
    )

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return X[indices, :], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    return np.linalg.solve(X.T @ X, X.T @ y)

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_pred - y_true))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    SS_res = np.sum((y_true - y_pred) ** 2)
    mean_y = np.mean(y_true)
    SS_tot = np.sum((y_true - mean_y) ** 2)
    if SS_tot == 0:
        return 0.0
    return 1 - SS_res / SS_tot

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r = y_true - y_pred
    return {
        "mean" : np.mean(r),
        "std" : np.std(r, ddof=0),
        "median_abs" : np.median(np.abs(r))
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    X = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X, k=iqr_k)
    return clip_columns(X, lower, upper)

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

