import numpy as np
import pytest
from chenchen_rl.linear_models import LinearRegression


def test_save_then_load_roundtrip(tmp_path):
    # Train a small model
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    w = np.array([1.5, -0.7])
    b = 0.25
    y = X @ w + b + rng.normal(scale=0.01, size=100)

    m1 = LinearRegression(use_intercept=True).fit(X, y)

    # Save
    path = tmp_path / "linreg_model.npz"
    m1.save(path)

    # Load
    m2 = LinearRegression.load(path)

    # Parameters close
    assert m2.use_intercept is True
    assert np.allclose(m1.coef_, m2.coef_, atol=1e-10)
    assert np.isclose(m1.intercept_, m2.intercept_, atol=1e-10)

    # Predictions identical
    X_test = rng.normal(size=(10, 2))
    y1 = m1.predict(X_test)
    y2 = m2.predict(X_test)
    assert np.allclose(y1, y2, atol=1e-12)


def test_save_raises_if_not_fitted(tmp_path):
    m = LinearRegression()
    with pytest.raises(RuntimeError):
        m.save(tmp_path / "bad.npz")


def test_load_rejects_wrong_class(tmp_path):
    # Create a bogus npz with different class_name
    bogus = tmp_path / "bogus.npz"
    np.savez_compressed(bogus, class_name="SomethingElse", use_intercept=True, coef=np.array([1.0]), intercept=0.0, fitted=True)
    with pytest.raises(ValueError):
        LinearRegression.load(bogus)
