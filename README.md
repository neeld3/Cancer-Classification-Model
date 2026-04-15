# Lung Cancer Prediction using Machine Learning

## Overview

This project predicts the likelihood of a patient having lung cancer (pulmonary disease) using healthcare data. It is a **binary classification problem** where multiple machine learning models are evaluated and compared.

We implemented and analyzed:

* Logistic Regression
* Support Vector Machines (SVM)
* Neural Networks (MLP)

The goal was to determine which model provides the best predictive performance.

---

## Dataset

* **Source:** Kaggle

* **Samples:** 5000

* **Train/Test Split:** 75% / 25% (3750 / 1250)

* **Features (17 total):**

  * Age, Gender
  * Smoking, Family Smoking History
  * Finger Discoloration
  * Mental Stress, Stress Immunity
  * Pollution Exposure
  * Long-term Illness
  * Energy Level, Immune Weakness
  * Breathing Issues, Chest Tightness
  * Alcohol Consumption
  * Throat Discomfort
  * Oxygen Saturation
  * Family History

* **Target Variable:**

  * Pulmonary Disease (Lung Cancer) → converted to binary (0/1)

---

## Exploratory Data Analysis

### Key Observations

* Most features are **binary**, limiting visual separability.
* **Age** is slightly right-skewed (more patients > 40).
* Several features (e.g., smoking, breathing issues) are skewed toward "YES".
* Correlation exists between:

  * Smoking ↔ Family Smoking History

### Dimensionality Reduction

* Applied **PCA + K-Means**
* Observed **clear clustering**, indicating separability in the dataset

---

## Data Preprocessing

* Randomized dataset to remove bias
* Converted categorical target → numeric (YES/NO → 1/0)
* Normalized features using **StandardScaler**
* Checked for:

  * Missing values → none
  * Outliers → none significant

---

## Models & Methods

### 1. Logistic Regression

* Tested:

  * No transformation
  * Polynomial features (degree 2–4)
* Regularization:

  * L1 and L2
  * C values from 0.0001 → 10000

#### Key Result

* **Best Model:** No transformation, no regularization
* **Test Accuracy:** 0.9096
* **F1 Score:** 0.8959

#### Insight

* Simpler models outperformed complex ones
* Polynomial features led to **overfitting**

---

### 2. Support Vector Machine (SVM)

* Kernels tested:

  * Linear
  * Polynomial
  * RBF
  * Sigmoid
* Used L2 regularization only

#### Key Result

* **Best Model:** RBF Kernel (C = 1)
* **Test Accuracy:** 0.8952
* **F1 Score:** 0.8779

#### Insight

* RBF performed best but showed **overfitting at high C**
* Sigmoid kernel underperformed consistently

---

### 3. Neural Networks (MLP)

Architectures tested:

* **Base:** (17 → 10 → 1)
* **Modified:** (17 → 20 → 10 → 1)
* **Wide:** (17 → 75 → 75 → 1)
* **Deep:** (17 → 50 → 40 → 30 → 1)

Regularization (L2 / alpha):

* 0 → 10000

#### Key Result

* **Best Model:** (17 → 20 → 10 → 1), alpha = 0.01
* **Test Accuracy:** **0.9216**
* **F1 Score:** ~0.908

#### Insight

* Moderate regularization improves generalization
* High regularization → **dead neurons / underfitting**
* Low regularization → **overfitting**

---

## Final Comparison

| Model               | Best Accuracy |
| ------------------- | ------------- |
| Logistic Regression | 0.9096        |
| SVM (RBF)           | 0.8952        |
| Neural Network      | **0.9216**    |

---

## Conclusion

* The **Neural Network (2 hidden layers)** performed the best overall.
* Logistic Regression was surprisingly competitive despite its simplicity.
* Increasing model complexity does **not always improve performance**.

---

## Future Improvements

* Feature selection (remove highly correlated features)
* Add new features:

  * Environmental exposure
  * Genetic risk factors
* Hyperparameter tuning with grid/random search
* Try ensemble methods (Random Forest, XGBoost)

---

## Tech Stack

* Python
* Scikit-learn
* NumPy / Pandas
* Matplotlib

---

## References

* Scikit-learn Documentation
* PCA, KMeans, Logistic Regression, SVM, MLPClassifier APIs
* https://scikit-learn.org/

---

## Authors

* Neel Dahake
* Gabriel Draghici
