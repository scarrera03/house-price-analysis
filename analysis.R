# House Price Analysis - Ames Housing Dataset

# 1. Required libraries
library(tidyverse)
library(caret)
library(rpart)
library(Metrics)

# 2. Load the dataset
houses <- read.csv("houses.csv")

# 3. Remove unnecessary columns
houses$Id <- NULL

# 4. Split into training (70%) and test (30%)
set.seed(123)
trainIndex <- createDataPartition(houses$SalePrice, p = 0.7, list = FALSE)
train <- houses[trainIndex, ]
test <- houses[-trainIndex, ]

# 5. Linear regression model
model_lm <- lm(SalePrice ~ ., data = train)
summary(model_lm)

# Linear regression prediction
predictions <- predict(model_lm, newdata = test)

# 6. Decision tree model
model_tree <- rpart(SalePrice ~ ., data = train, method = "anova")
pred_tree <- predict(model_tree, newdata = test)

# 7. Evaluation: Mean Squared Error (MSE)
mse_lm <- mse(test$SalePrice, predictions)
mse_tree <- mse(test$SalePrice, pred_tree)

# 8. Results
cat("MSE - Linear Regression:", mse_lm, "\n")
cat("MSE - Decision Tree:", mse_tree, "\n")

# 9. Visual comparison
df <- data.frame(
  actuals = test$SalePrice,
  pred_lm = predictions,
  pred_tree = pred_tree
)

ggplot(df, aes(x = actuals)) +
  geom_point(aes(y = pred_lm, color = "Linear Regression"), alpha = 0.6) +
  geom_point(aes(y = pred_tree, color = "Decision Tree"), alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Comparison of Predictions vs Actual Values",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal() +
  scale_color_manual(values = c("Linear Regression" = "blue", "Decision Tree" = "darkgreen"))
