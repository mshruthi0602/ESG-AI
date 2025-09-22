# src/shap_utils.py
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_shap_summary(model, X_val, feature_names, outpath="outputs/shap_summary.png"):
    """
    Works for tree-based models that shap.TreeExplainer supports (RF, XGB, LGBM).
    Produces a standard SHAP summary (beeswarm) image for the appendix.
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    try:
        # RF returns list; XGB/LGBM often return array
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    except Exception:
        sv = shap_values
    shap.summary_plot(sv, X_val, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
