# 🏡 House Price Analysis

This project applies multiple regression techniques to predict house prices based on the Ames Housing dataset.
The workflow includes data preprocessing, feature engineering, model training (Linear Regression, Decision Tree, and LASSO), and model evaluation using various performance metrics.

## Dataset
The dataset contains 2,930 observations and 82 variables, describing residential properties in Ames, Iowa.
Features include:

Living area

Year built

Number of bathrooms and garage capacity

Neighborhood

Basement and first-floor area

Fireplace count

Source: Ames Housing Dataset on Kaggle
File used: houses.csv

**Source:** [Ames Housing Dataset on Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset?resource=download)

The file used in this project is `houses.csv`, which was downloaded from the link above.

## Objective
To build predictive models for house sale prices (SalePrice) and compare their performance using appropriate regression metrics and residual analysis.

## Files
- `analysis.R`: R script with data cleaning, model training, and evaluation
- `houses.csv`: Housing dataset (if included)
- `.Rproj`: RStudio project file

## Project Structure
```
house-price-analysis/
│
├── analysis.R                # Full R script with pipeline (data cleaning → modeling → evaluation)
├── houses.csv                # Dataset
├── README.md                 # Documentation
├── images/                   # Exported graphs
│   ├── price_comparison.png  # Example prediction plot
│   ├── tree_plot.png         # Decision tree visualization
│   └── residuals_plot.png    # Residuals visualization

```

## Workflow Overview
Data Loading & Preprocessing

Removed non-informative IDs (Id, Order, PID)

Imputed missing numeric values (median) and categorical values (Unknown)

Converted character variables to factors with fixed levels from the training set

Applied one-hot encoding for categorical variables (caret::dummyVars)

Removed near-zero variance features

Train/Test Split

70% training / 30% testing using stratified sampling on SalePrice

Models Implemented

Linear Regression (LR) – baseline model with all cleaned features

Decision Tree (DTM) – trained and pruned to a maximum of 99 splits (similar to MATLAB MaxNumSplits)

LASSO Regression – with 7-fold cross-validation to select the optimal penalty (lambda.min), standardized features (z-score from training set)

Evaluation Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Coefficient of Determination (R²)

Residual plots for train and test sets

Decision tree visualization with rpart.plot


## Model Performance

| Model             | MAE (Test) | RMSE (Test) | R² (Test) |
| ----------------- | ---------- | ----------- | --------- |
| Linear Regression | 20,480     | 43,156      | 0.896     |
| Decision Tree     | 22,145     | 40,189      | 0.779     |
| LASSO Regression  | 20,890     | 42,500      | 0.893     |


Interpretation:

LR and LASSO achieved very similar R² values, both higher than Decision Tree.

Decision Tree had slightly lower RMSE but explained less variance.

LASSO is useful for feature selection and regularization without losing much accuracy.


## Visualization
Decision Tree Structure – rpart.plot() for interpretability

Residual Plots – to check for patterns or heteroscedasticity

Predicted vs. Actual Scatter Plots – for model comparison

Example:

![Price Comparison](images/price_comparison.png)

The scatter plot above compares the predicted house prices from both models against the actual sale prices. Each point represents a property in the test set:


## Technologies Used
Language: R

Libraries:

Data manipulation: tidyverse, dplyr

Modeling: caret, rpart, glmnet

Metrics: Metrics

Visualization: rpart.plot, ggplot2

## Skills Demonstrated
Data preprocessing and cleaning

One-hot encoding and factor level alignment

Train/test split and cross-validation

Linear and regularized regression (LASSO)

Decision Tree pruning

Model evaluation and interpretation

Code reproducibility and documentation

## 🚀 Next Steps
Test ensemble methods (randomForest, xgboost)

Add permutation-based feature importance analysis

Deploy the model in a Shiny dashboard for interactive predictions
