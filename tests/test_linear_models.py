import numpy as np
import pytest
from chenchen_rl.linear_models import LinearRegression

def test_fit_predict_with_intercept():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    w = np.array([1.5, -2.0])
    b = 0.7
    y = X @ w + b + rng.normal(scale=0.05, size=200)

    lr = LinearRegression(use_intercept=True).fit(X, y)
    y_hat = lr.predict(X)

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.98
    assert lr.coef_.shape == (2,)
    assert abs(lr.intercept_ - b) < 0.2

def test_fit_predict_no_intercept():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 1))
    w = np.array([3.0])
    y = X @ w + rng.normal(scale=0.05, size=150)

    lr = LinearRegression(use_intercept=False).fit(X, y)
    y_hat = lr.predict(X)
    assert np.corrcoef(y, y_hat)[0, 1] > 0.99
    assert lr.intercept_ == 0.0
    assert lr.coef_.shape == (1,)

def test_predict_requires_fit():
    lr = LinearRegression()
    with pytest.raises(RuntimeError):
        lr.predict([1.0])
