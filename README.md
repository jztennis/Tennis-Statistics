# 🧠 Custom Tennis Machine Learning Framework (Python)

A modular, from-scratch machine learning toolkit for classification, clustering, and data analysis using a custom table abstraction. Built as part of an academic project to understand the inner workings of popular ML algorithms without relying on external machine learning libraries.

---

## 📦 Project Structure
```bash
├── data_table.py # Core table abstraction (DataTable, DataRow) with CSV support
├── data_util.py # Utility functions: normalization, discretize, mean, variance, etc.
├── data_learn.py # Machine learning algorithms (Naive Bayes, k-NN, Gaussian Density, K-Means, etc)
├── data_eval.py # Evaluation functions and helpers: tdidt, random forest eval, stratify, union all, etc.
├── decision_tree.py # Trees: LeadfNode, AttributeNode, draw_tree.
├── match_scores_stats_2017_setup_csv.csv # data
├── README.md # This file
```
---

## ✅ Features

### 📊 Table Management
- Lightweight `DataTable` and `DataRow` classes (similar to pandas)
- Support for row filtering, projection, joining, and duplicate removal
- CSV loading and saving for structured tabular data

### 🤖 Machine Learning
- **Naive Bayes Classifier** (supports both continuous and categorical features)
- **k-Nearest Neighbors (k-NN)** with majority and weighted voting
- **Gaussian Density Estimation** for probabilistic classification
- **Decision Tree Learning (TDIDT)** with information gain and pruning
- **K-Means Clustering** for unsupervised learning
- **Random Forest Evaluation** using randomized feature selection and voting

### 🧪 Model Evaluation
- Stratified sampling and train-test splitting
- Accuracy and confusion matrix generation
- Cross-validation with multiple random trials

### 📐 Utilities
- Column normalization and discretization
- Summary statistics: mean, standard deviation, variance
- Correlation, linear regression, and scatter plotting
- Visualizations: histograms, pie charts, bar graphs, box plots

---
