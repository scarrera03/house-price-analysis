# 🏡 House Price Analysis

This project uses regression techniques to predict house prices based on various features from a structured dataset. It compares a Linear Regression model and a Decision Tree model, evaluating their performance on a hold-out test set using the Ames Housing dataset.

## Dataset
The dataset contains 2,930 observations and 82 variables, describing residential properties in Ames, Iowa. It includes features such as:

- Living area
- Year built
- Number of bathrooms and garage capacity
- Neighborhood
- Basement and first-floor area
- Fireplace count

**Source:** [Ames Housing Dataset on Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset?resource=download)

The file used in this project is `houses.csv`, which was downloaded from the link above.

## Objective
The goal is to build models that can accurately predict house sale prices (SalePrice) and evaluate their performance using appropriate regression metrics.

## Files
- `analysis.R`: R script with data cleaning, model training, and evaluation
- `houses.csv`: Housing dataset (if included)
- `.Rproj`: RStudio project file

## Project Structure
```
house-price-analysis/
│
├── analysis.R                # R script with full pipeline
├── house_price_analysis.Rmd  # Optional R Markdown notebook
├── houses.csv                # Dataset
├── README.md                 # Project documentation
├── images/                   # Exported graphs
│   └── price_comparison.png  # Saved ggplot image
```

## Workflow Overview
1. **Data loading and preprocessing**
   - Removed ID variables (`Id`, `Order`, `PID`)
   - Converted character variables to factors
   - Removed low-variance and low-level categorical features
   - Imputed missing values (median for numeric, mode for categorical)

2. **Train/test split**
   - Stratified split: 70% training, 30% testing (`caret::createDataPartition`)

3. **Model training**

- Linear Regression using all cleaned features (`lm(SalePrice ~ ., data = train_clean)`)
- Decision Tree Regression trained on the original training set (`rpart(SalePrice ~ ., data = train, method = "anova")`)

4. **Evaluation**

Model performance was evaluated using:

- MSE (Mean Squared Error) on the test set
- Visual comparison of predicted vs. actual sale prices using scatter plots


## Model Performance

| Model             | RMSE       | R²     | MSE         |
|------------------|------------|--------|-------------|
| Linear Regression| 43,156.42  | 0.8955 | 1.86e+09    |
| Decision Tree    | 40,189.34  | 0.7792 | 1.62e+09    |

> While the Decision Tree model achieved a slightly lower MSE, the Linear Regression model had a significantly higher R², meaning it explained more of the variance in house prices. This suggests that, overall, linear regression provides more reliable predictions despite slightly higher absolute error.


## Visualization
A regression tree was plotted using rpart.plot() for interpretability

Predicted vs. Actual values can be visualized with ggplot2

## 📷 Model Predictions Visualization

![Price Comparison](images/price_comparison.png)

The scatter plot above compares the predicted house prices from both models against the actual sale prices. Each point represents a property in the test set:

- **Blue dots** correspond to predictions from the **Linear Regression model**
- **Green dots** correspond to predictions from the **Decision Tree model**
- The **black dashed line** represents perfect prediction (i.e., predicted = actual)

We observe that the linear regression predictions align more closely to the ideal line, indicating better accuracy overall. In contrast, the decision tree model shows more horizontal clustering, which is typical of trees that predict in discrete value steps. This visual reinforces the evaluation metrics: the linear regression model achieved lower RMSE and higher R² compared to the decision tree.

## Models
- Linear Regression
- Decision Tree

## Metrics
- Mean Squared Error (MSE)
- Visual comparison of predictions vs actual prices

## Technologies Used
- **Language**: R
- **Libraries**: `tidyverse`, `caret`, `rpart`, `rpart.plot`, `Metrics`, `ggplot2`, `dplyr`

## Skills Demonstrated
- Data cleaning and preprocessing
- Factor level alignment across datasets
- Model training and hyperparameter-free comparison
- Visualization of results using `ggplot2`
- Regression model evaluation using MSE
- Project documentation and reproducibility

## 🚀 Next Steps

- Try ensemble methods (`randomForest`, `xgboost`)
- Add variable importance and residual plots
- Deploy as a simple Shiny web app for user interaction
