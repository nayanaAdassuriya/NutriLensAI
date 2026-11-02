"""
Explainable AI (XAI) Module for NutriLens AI.

This module provides post-hoc interpretability for the CatBoost Classifier using
SHAP (SHapley Additive exPlanations) TreeExplainer:
1. Global feature importance analysis revealing high-level model mechanics.
2. Local feature attribution for individual food products via SHAP waterfall plots.
3. Domain-aligned natural language explanation translating shapley values into
   actionable nutritional insights.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier
from data_pipeline import load_and_preprocess_dataset, ARTIFACTS_DIR
from train_model import MODEL_PATH


SHAP_SUMMARY_PATH = os.path.join(ARTIFACTS_DIR, "shap_summary.png")


class NutriLensExplainer:
    """
    Wrapper for SHAP TreeExplainer designed specifically for the CatBoost multi-class
    nutritional grading model.
    """
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model artifact not found at {model_path}. Train the model first.")
            
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        
        # Load feature and class metadata
        _, self.X_test, _, _, self.feature_names, self.metadata = load_and_preprocess_dataset()
        self.class_names = self.metadata["class_names"]

    def generate_global_explanations(self, max_samples: int = 200, save_path: str = SHAP_SUMMARY_PATH):
        """
        Computes SHAP values over a representative test sample and generates
        a global summary beeswarm/bar plot for reporting.
        """
        sample_subset = self.X_test.iloc[:max_samples]
        shap_values = self.explainer(sample_subset)
        
        plt.figure(figsize=(9, 6))
        # For multi-class, shap_values has shape (samples, features, classes)
        # We can take mean absolute shap values across classes or plot primary grade E/A
        if len(shap_values.shape) == 3:
            # Multi-class: plot summary for Class E (unhealthy) or mean importance
            shap.summary_plot(
                shap_values[:, :, 4].values,
                sample_subset,
                feature_names=self.feature_names,
                show=False
            )
            plt.title("SHAP Feature Attribution (Impact on Class E - Unhealthy Grade)", fontsize=11, pad=12)
        else:
            shap.summary_plot(shap_values, sample_subset, show=False)
            plt.title("SHAP Feature Attribution Summary", fontsize=11, pad=12)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[XAI] Global SHAP summary plot saved to {save_path}")
        return shap_values

    def explain_instance(self, sample_df: pd.DataFrame) -> dict:
        """
        Generates local explanation for a single packaged food item.
        Returns:
            - predicted_class: str ('A' through 'E')
            - class_probabilities: dict of grade -> prob
            - shap_values: array of feature attributions for predicted class
            - textual_insights: list of human-readable nutritional drivers
            - base_value: expected baseline log-odds/value
        """
        # Ensure column order matches training
        sample_ordered = sample_df[self.feature_names].copy()
        
        # Get model prediction and probability distribution
        probabilities = self.model.predict_proba(sample_ordered)[0]
        pred_idx = int(np.argmax(probabilities))
        pred_grade = self.class_names[pred_idx]
        
        # Compute SHAP explanation for the instance
        shap_explanation = self.explainer(sample_ordered)
        
        # Multi-class indexing: shape is (1, n_features, n_classes)
        if len(shap_explanation.shape) == 3:
            class_shap_vals = shap_explanation.values[0, :, pred_idx]
            base_val = shap_explanation.base_values[0, pred_idx]
        else:
            class_shap_vals = shap_explanation.values[0]
            base_val = shap_explanation.base_values[0]
            
        # Compile textual insights explaining positive/negative forces
        feature_impacts = list(zip(self.feature_names, class_shap_vals, sample_ordered.iloc[0].values))
        # Sort by absolute SHAP impact
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        textual_insights = []
        for feat, sh_val, feat_val in feature_impacts[:4]:
            direction = "increased" if sh_val > 0 else "decreased"
            status = "driving" if sh_val > 0 else "counteracting"
            
            # Map human biological explanation
            if "sugar" in feat:
                desc = f"Sugar content ({feat_val:.1f}g/100g) {status} the likelihood of Grade {pred_grade} by {abs(sh_val):.2f} log-odds."
            elif "sodium" in feat:
                desc = f"Sodium level ({feat_val:.1f}mg/100g) {status} classification into Grade {pred_grade} (impact: {sh_val:+.2f})."
            elif "saturated_fat" in feat:
                desc = f"Saturated fatty acid content ({feat_val:.1f}g/100g) {status} Grade {pred_grade} (impact: {sh_val:+.2f})."
            elif "fiber" in feat:
                desc = f"Dietary fiber ({feat_val:.1f}g/100g) acted as a beneficial factor (impact: {sh_val:+.2f})."
            elif "proteins" in feat:
                desc = f"Protein density ({feat_val:.1f}g/100g) influenced the rating (impact: {sh_val:+.2f})."
            elif "additives" in feat:
                desc = f"Presence of {int(feat_val)} food additives/E-numbers contributed {sh_val:+.2f} toward the grade."
            else:
                desc = f"Feature '{feat}' (value: {feat_val:.1f}) {direction} the grade log-odds by {abs(sh_val):.2f}."
            textual_insights.append(desc)
            
        prob_dist = {self.class_names[i]: float(probabilities[i]) for i in range(len(self.class_names))}
        
        return {
            "predicted_grade": pred_grade,
            "predicted_index": pred_idx,
            "probabilities": prob_dist,
            "shap_values": class_shap_vals,
            "base_value": float(base_val),
            "feature_names": self.feature_names,
            "feature_values": sample_ordered.iloc[0].values,
            "textual_insights": textual_insights,
            "shap_explanation": shap_explanation
        }

    def plot_local_waterfall(self, explanation_result: dict, title: str = "Local Feature Attribution"):
        """
        Creates a clean horizontal bar figure representing the SHAP feature contributions
        for embedding inside the Streamlit frontend.
        """
        feats = explanation_result["feature_names"]
        sh_vals = explanation_result["shap_values"]
        f_vals = explanation_result["feature_values"]
        
        labels = [f"{feat} = {val:.1f}" for feat, val in zip(feats, f_vals)]
        
        # Sort by magnitude of shapley values
        sorted_indices = np.argsort(np.abs(sh_vals))
        
        fig, ax = plt.subplots(figsize=(8, 4.8))
        colors = ["#22c55e" if val < 0 else "#ef4444" for val in sh_vals[sorted_indices]]
        
        ax.barh(
            [labels[i] for i in sorted_indices],
            sh_vals[sorted_indices],
            color=colors,
            alpha=0.85,
            edgecolor="#1e293b",
            linewidth=0.8
        )
        
        ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"SHAP Value (Impact on Grade {explanation_result['predicted_grade']} Log-Odds)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        
        # Annotate values
        for idx, (sh, lbl) in enumerate(zip(sh_vals[sorted_indices], [labels[i] for i in sorted_indices])):
            offset = 0.02 if sh >= 0 else -0.02
            ha = 'left' if sh >= 0 else 'right'
            ax.text(sh + offset, idx, f"{sh:+.2f}", va='center', ha=ha, fontsize=9, fontweight='semibold')
            
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    explainer = NutriLensExplainer()
    print("[XAI] Generating global explanation beeswarm...")
    explainer.generate_global_explanations()
    
    # Test on a single sample (e.g. sugary biscuit)
    test_sample = pd.DataFrame([{
        "energy_kcal": 490.0,
        "sugars_g": 31.5,
        "fat_g": 21.0,
        "saturated_fat_g": 11.2,
        "sodium_mg": 280.0,
        "proteins_g": 5.2,
        "fiber_g": 1.2,
        "fruits_veg_pct": 0.0,
        "is_beverage": 0,
        "additives_count": 4
    }])
    
    res = explainer.explain_instance(test_sample)
    print(f"\nPredicted Grade: {res['predicted_grade']}")
    print("Probabilities:", res["probabilities"])
    print("Insights:")
    for ins in res["textual_insights"]:
        print(" -", ins)
