"""Competition metric: |y|-weighted accuracy of the predicted sign. A prediction of 0 is always wrong."""
import numpy as np


def weighted_sign_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.abs(y_true)
    hit = (np.sign(y_pred) == np.sign(y_true)) & (y_pred != 0)
    return float((w * hit).sum() / w.sum())
