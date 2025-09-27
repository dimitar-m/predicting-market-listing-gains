# TensorFlow Functional API Model for Listing-Gain Prediction in Indian IPOs

This repo builds a compact neural network (TensorFlow Functional API) to estimate the probability that an Indian IPO delivers a **listing-day gain**. The workflow covers EDA, preprocessing, stratified splitting, reproducible training, and validation-based threshold selection.

## Highlights
- **Model**: 7 numeric inputs → Dense(16, ReLU) → BatchNorm → Dropout(0.2) → Dense(8, ReLU) → BatchNorm → Dropout(0.2) → Sigmoid.
- **Regularization**: L2 on Dense layers + Dropout + BatchNorm.
- **Reproducibility**: fixed seeds, deterministic stratified split, `shuffle=False`.
- **Thresholding**: decision threshold chosen on validation by **max F1**.

## Data & Splits
- Target: **listing-day gain** (1) vs **no gain** (0).
- Inputs: 7 engineered/numeric features (see notebook).
- Sizes (stratified): Train **204**, Val **51**, Test **64**.

## Results
- **Validation**: ROC-AUC **0.744**, PR-AUC **0.773**, best F1 **0.765** @ **t = 0.32**.
- **Test**: ROC-AUC **0.725**; high recall for gains with a trade-off in specificity at the chosen threshold.
- Plots: Validation ROC and PR curves included in the notebook.

## Methodology
1. **EDA**: data quality checks, distributions, correlations.
2. **Preprocessing**: float32 casting, sanity checks, stratified split with fixed seed.
3. **Modeling**: small MLP (Functional API) with Adam, EarlyStopping, and ReduceLROnPlateau.
4. **Evaluation**: ROC/PR curves, AUCs, F1-optimized threshold on validation; test metrics reported at that threshold.

## Environment (conda)
> GPU optional. Uses **TensorFlow 2.17.0** with **NumPy 1.26.4** to avoid ABI issues.

```bash
conda create -n tf-gpu-123 python=3.10 -y
conda activate tf-gpu-123

# Core scientific stack
conda install -y numpy=1.26.4 pandas=2.3.2 scikit-learn=1.5.2 matplotlib=3.8.4 seaborn=0.13.2

# TensorFlow (CPU or your GPU build)
pip install tensorflow==2.17.0

# Recommended: avoid user-site shadowing when launching Jupyter
export PYTHONNOUSERSITE=1
python -m jupyter lab
Reproducibility Notes
Seeds set for NumPy and TF; stratified split uses fixed random_state.

model.fit(..., shuffle=False) and a saved initial weight snapshot ensure deterministic re-runs in the same process.

How to Run
Open the notebook in Jupyter Lab.

Execute cells in order: EDA → preprocessing → model build → training → evaluation.

Figures (ROC/PR) are saved alongside the notebook when executed.

License
Released under the MIT License (see LICENSE).
