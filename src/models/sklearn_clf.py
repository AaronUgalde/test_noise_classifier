"""Scikit-learn classifiers for feature-based models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def build_classifier(clf_type: str, config: dict, seed: int):
    if clf_type == "lr":
        lr_cfg = config["sklearn"]["logistic_regression"]
        return LogisticRegression(
            C=lr_cfg["C"],
            max_iter=lr_cfg["max_iter"],
            class_weight=lr_cfg["class_weight"],
            solver=lr_cfg.get("solver", "lbfgs"),
            random_state=seed,
        )
    if clf_type == "svm":
        svm_cfg = config["sklearn"]["linear_svm"]
        base = LinearSVC(
            C=svm_cfg["C"],
            class_weight=svm_cfg["class_weight"],
            max_iter=svm_cfg["max_iter"],
            random_state=seed,
            dual="auto",
        )
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(f"Unknown classifier type: {clf_type}")


def train_predict_sklearn(clf, x_train, y_train, x_test) -> tuple[np.ndarray, np.ndarray]:
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(x_test)[:, 1]
    else:
        y_prob = clf.decision_function(x_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
    return y_pred.astype(int), y_prob.astype(float)
