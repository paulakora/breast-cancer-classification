# Breast Cancer Classification with Machine Learning

## Project Overview
This project aims to classify breast cancer tumors as malignant (M) or benign (B) using various supervised machine learning models. It demonstrates the full pipeline from data preprocessing to model evaluation.

## Dataset
- **Source:** `Cancer_Data.csv`
- **Samples:** 569
- **Features:** 30 numeric diagnostic attributes (e.g., radius, texture, area)
- **Target Variable:** `diagnosis` (Malignant `M`, Benign `B`)

## Data Preprocessing
- Removed irrelevant columns (`id`, unnamed column).
- Verified data cleanliness (no missing or duplicated values).
- Standardized features with `StandardScaler`.
- Visualized feature distributions and class balance.
- Used **PCA (2 components)** for data visualization and dimensionality insight.

## Exploratory Data Analysis
- Analyzed feature distributions for skewness.
- Plotted correlation heatmap to spot multicollinearity.
- Observed class imbalance: ~63% benign, ~37% malignant.

## Models Used
All models were evaluated using accuracy, precision, recall, F1-score, and confusion matrix:

| Model                  | Notes                                     |
|------------------------|-------------------------------------------|
| `DummyClassifier`      | Baseline (most frequent class)            |
| `DecisionTreeClassifier` | Grid search over depth, splits, and leaves |
| `LogisticRegression`   | L1/L2 regularization, multiple C values   |
| `KNeighborsClassifier` | Tuned k-values and weighting strategies   |
| `SVC (Support Vector Machine)` | Linear and RBF kernels evaluated       |

## Results
- PCA confirmed the dataset is linearly separable to a good extent.
- Logistic Regression and SVM achieved strong performance.
- All tuned models significantly outperformed the baseline.

## Evaluation Metrics
- Classification reports and confusion matrices visualized.
- Cross-validation and `GridSearchCV` used for model selection.

## Key Learnings
- How to build an end-to-end ML pipeline.
- Importance of baseline comparison.
- Role of scaling and PCA in improving model effectiveness.
- Value of hyperparameter tuning with grid search.

## Next Steps
- Introduce ensemble models (e.g., Random Forest, XGBoost).
- Add feature selection or dimensionality reduction techniques.
- Apply resampling techniques (e.g., SMOTE) for improved recall.

---

**Author:** *Paula Koralewska*  
**Contact:** *work.paula.koralewska@gmail.com*  
**License:** MIT
