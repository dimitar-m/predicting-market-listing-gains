# TensorFlow Functional API Model for Listing-Gain Prediction in Indian IPOs

This repo builds a compact neural network (TensorFlow Functional API) to estimate the probability that an Indian IPO delivers a **listing-day gain**. The workflow covers EDA, preprocessing, stratified splitting, reproducible training, and validation-based threshold selection.

## How to Run  
- Open the latest version of the project in **NBViewer**:  
  [View in NBViewer](https://nbviewer.org/github/dimitar-m/predicting-market-listing-gains/blob/master/predicting-listing-gains.ipynb)
  
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

## License
- Released under the MIT License (see LICENSE).
