from __future__ import annotations
from pathlib import Path
import numpy as np


class LinearRegression:
    """
    Ordinary Least Squares using the normal equation with pseudo-inverse.

    Parameters
    ----------
    use_intercept : bool, default=True
        If True, fit an intercept term (bias). If False, force intercept = 0.
    """

    def __init__(self, use_intercept: bool = True) -> None:
        self.use_intercept = bool(use_intercept)
        self.coef_: np.ndarray | None = None    # shape (n_features,)
        self.intercept_: float | None = None
        self._fitted: bool = False

    @staticmethod
    def _check_X_y(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Ensure 2D X and 1D y
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim != 1:
            y = y.ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of rows: {X.shape[0]} vs {y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("X and y must be non-empty.")
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit model parameters by solving (X^T X) w = X^T y via pseudo-inverse.
        """
        X, y = self._check_X_y(X, y)

        if self.use_intercept:
            ones = np.ones((X.shape[0], 1), dtype=X.dtype)
            X_aug = np.hstack([ones, X])
        else:
            X_aug = X

        theta = np.linalg.pinv(X_aug) @ y  # robust to singular X^T X

        if self.use_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = theta.copy()

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for X.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit(X, y) first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_hat = X @ self.coef_
        if self.use_intercept:
            y_hat = y_hat + self.intercept_
        return y_hat

    # ---------- NEW: persistence ----------
    def save(self, filepath: str | Path) -> None:
        """
        Save the fitted model to a compressed .npz file.

        Parameters
        ----------
        filepath : str or Path
            Destination path (e.g., 'model.npz').
        """
        if not self._fitted or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Cannot save an unfitted model. Call fit(X, y) first.")

        filepath = Path(filepath)
        np.savez_compressed(
            filepath,
            class_name="LinearRegression",
            version="0.1",
            use_intercept=np.array(self.use_intercept, dtype=bool),
            coef=self.coef_.astype(np.float64, copy=False),
            intercept=np.array(self.intercept_, dtype=np.float64),
            fitted=np.array(self._fitted, dtype=bool),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> "LinearRegression":
        """
        Load a model saved with `save`.

        Parameters
        ----------
        filepath : str or Path
            Path to the .npz file.

        Returns
        -------
        LinearRegression
            A fitted LinearRegression instance.
        """
        filepath = Path(filepath)
        with np.load(filepath, allow_pickle=False) as data:
            # Optional sanity checks
            if "class_name" in data and str(data["class_name"]) != "LinearRegression":
                raise ValueError("File does not contain a LinearRegression model.")
            use_intercept = bool(np.asarray(data["use_intercept"]).item())
            coef = np.asarray(data["coef"], dtype=np.float64)
            intercept = float(np.asarray(data["intercept"], dtype=np.float64))
            fitted = bool(np.asarray(data["fitted"]).item())

        model = cls(use_intercept=use_intercept)
        model.coef_ = coef
        model.intercept_ = intercept
        model._fitted = fitted
        if not model._fitted:
            # In case someone tampers with the file
            raise ValueError("Loaded model is not marked as fitted.")
        return model
