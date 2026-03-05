import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score


def select_binary_threshold_with_constraints(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    *,
    acc_min: float = 0.80,
    prec_min: float = 0.67,
    objective: str = "recall",
    t_min: float = 0.20,
    t_max: float = 0.80,
    n_points: int = 301,
):
    thresholds = np.linspace(t_min, t_max, n_points)
    candidates = []

    for t in thresholds:
        y_pred = (p_pos >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bacc = balanced_accuracy_score(y_true, y_pred)

        if acc >= acc_min and prec >= prec_min:
            candidates.append(
                {
                    "threshold": float(t),
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "balanced_accuracy": float(bacc),
                }
            )

    if not candidates:
        return None, []

    objective = objective.lower().strip()
    if objective == "balanced_accuracy":
        key_fn = lambda r: (r["balanced_accuracy"], r["f1"], r["recall"])
    elif objective == "f1":
        key_fn = lambda r: (r["f1"], r["balanced_accuracy"], r["recall"])
    else:
        key_fn = lambda r: (r["recall"], r["balanced_accuracy"], r["f1"])

    best = sorted(candidates, key=key_fn, reverse=True)[0]
    return best, candidates
