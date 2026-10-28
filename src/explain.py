"""SHAP and LIME explanations for a single predicted session."""

import numpy as np
import pandas as pd


def get_shap_explainer(model):
    """Build a SHAP explainer for a tree model (LightGBM, XGBoost, forests...)."""
    import shap

    return shap.TreeExplainer(model)


def shap_contributions(explainer, row):
    """How much each feature pushed this prediction up (+) or down (-).

    Returns (table sorted by size of effect, the average prediction the push starts from).
    """
    explanation = explainer(row)
    values, base_values = explanation.values, explanation.base_values
    if values.ndim == 3:  # some versions return one layer per class
        values, base_values = values[:, :, 1], base_values[:, 1]

    table = pd.DataFrame(
        {
            "Feature": list(row.columns),
            "Value": row.iloc[0].values,
            "Contribution": values[0],
        }
    )
    order = table["Contribution"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True), float(np.ravel(base_values)[0])


def build_lime_explainer(training_features, mode):
    """LIME needs a sample of training data to learn what 'typical' looks like."""
    from lime.lime_tabular import LimeTabularExplainer

    names = list(training_features.columns)
    binary_positions = [
        i for i, name in enumerate(names) if training_features[name].nunique() <= 2
    ]
    return LimeTabularExplainer(
        training_data=training_features.values,
        feature_names=names,
        categorical_features=binary_positions,
        class_names=["Normal", "High"] if mode == "classification" else None,
        mode=mode,
        random_state=42,
    )


def lime_contributions(lime_explainer, predict_function, row, number_of_features=8):
    """The simple rules LIME found around this one prediction."""
    result = lime_explainer.explain_instance(
        row.iloc[0].values, predict_function, num_features=number_of_features, num_samples=2000
    )
    return pd.DataFrame(result.as_list(), columns=["Rule", "Effect"])
