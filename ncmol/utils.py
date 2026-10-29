from __future__ import annotations

import random

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or not len(y_true):
        raise ValueError("Metrics require equally sized nonempty 1D arrays")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics require finite targets and predictions")
    can_correlate = len(y_true) > 1 and np.ptp(y_true) > 0 and np.ptp(y_pred) > 0
    comparable = 0
    concordant = 0.0
    for index in range(len(y_true) - 1):
        delta_true = y_true[index + 1:] - y_true[index]
        mask = delta_true != 0
        if not np.any(mask):
            continue
        delta_pred = y_pred[index + 1:][mask] - y_pred[index]
        products = delta_true[mask] * delta_pred
        comparable += int(mask.sum())
        concordant += float((products > 0).sum())
        concordant += 0.5 * float((products == 0).sum())
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "pearson": float(pearsonr(y_true, y_pred).statistic) if can_correlate else float("nan"),
        "spearman": float(spearmanr(y_true, y_pred).statistic) if can_correlate else float("nan"),
        "ci": float(concordant / comparable) if comparable else float("nan"),
    }
