# ==========================
# House Price Analysis - Ames
# ==========================

# Packages (install them only once outside the script if missing)
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(Metrics)

# Added-variable plots (optional)
has_car <- requireNamespace("car", quietly = TRUE)
if (has_car) library(car)

# --- 1) Load dataset ---
houses <- read.csv("/Users/silvinaguadalupecarrerascholz/house-price-analysis/houses.csv")
stopifnot("SalePrice" %in% names(houses))

# --- 2) Holdout 70/30 ---
set.seed(123)
idx <- createDataPartition(houses$SalePrice, p = 0.7, list = FALSE)
train <- houses[idx, ]
test  <- houses[-idx, ]

# --- 3) Remove IDs that do not contribute (use only if present) ---
train <- train %>% select(-any_of(c("Id","Order","PID")))
test  <- test  %>% select(-any_of(c("Id","Order","PID")))

# --- 4) Imputation and consistent typing (similar to MATLAB) ---
# Ensure SalePrice has no NA values
train <- train[!is.na(train$SalePrice), ]
test  <- test[!is.na(test$SalePrice), ]

num_cols  <- names(which(sapply(train, is.numeric)))
char_cols <- names(which(sapply(train, is.character)))
fact_cols <- names(which(sapply(train, is.factor)))

# a) Numerics → median (from train)
for (nm in num_cols) {
  med <- suppressWarnings(median(train[[nm]], na.rm = TRUE))
  if (is.finite(med)) {
    train[[nm]][is.na(train[[nm]])] <- med
    if (nm %in% names(test)) test[[nm]][is.na(test[[nm]])] <- med
  } else {
    # Column entirely NA: remove in both
    train[[nm]] <- NULL
    if (nm %in% names(test)) test[[nm]] <- NULL
  }
}

# b) Categoricals → "Unknown" and fixed levels from train
to_factor <- union(char_cols, fact_cols)
for (nm in to_factor) {
  train[[nm]] <- as.character(train[[nm]])
  train[[nm]][is.na(train[[nm]]) | trimws(train[[nm]]) == ""] <- "Unknown"
  levs <- unique(train[[nm]])
  train[[nm]] <- factor(train[[nm]], levels = levs)
  
  if (nm %in% names(test)) {
    test[[nm]] <- as.character(test[[nm]])
    test[[nm]][is.na(test[[nm]]) | trimws(test[[nm]]) == ""] <- "Unknown"
    test[[nm]][!test[[nm]] %in% levs] <- "Unknown"
    test[[nm]] <- factor(test[[nm]], levels = levs)
  }
}

# --- 5) One-hot encoding (equivalent to MATLAB's onehotencode) ---
dmy <- dummyVars(SalePrice ~ ., data = train, fullRank = TRUE, na.action = na.pass)
Xtr <- data.frame(predict(dmy, newdata = train))
Xte <- data.frame(predict(dmy, newdata = test))

# Align columns between train and test (same features and order)
missing_in_test <- setdiff(names(Xtr), names(Xte))
if (length(missing_in_test) > 0) Xte[missing_in_test] <- 0
extra_in_test <- setdiff(names(Xte), names(Xtr))
if (length(extra_in_test) > 0) Xte <- Xte[, setdiff(names(Xte), extra_in_test), drop = FALSE]
Xte <- Xte[, names(Xtr), drop = FALSE]

train_df <- cbind(SalePrice = train$SalePrice, Xtr)
test_df  <- cbind(SalePrice = test$SalePrice,  Xte)

# (Optional) remove near-zero variance
nzv <- nearZeroVar(train_df[, -1, drop = FALSE])
if (length(nzv) > 0) {
  keep_idx <- setdiff(seq_len(ncol(train_df) - 1), nzv)
  sel <- c(1, 1 + keep_idx)
  train_df <- train_df[, sel, drop = FALSE]
  test_df  <- test_df[, names(train_df), drop = FALSE]
}

# ==========================
# 6) Linear Regression (LR)
# ==========================
model_lm  <- lm(SalePrice ~ ., data = train_df)

# Predictions
lrmPredTrain <- as.vector(predict(model_lm, newdata = train_df))
lrmPredTest  <- as.vector(predict(model_lm,  newdata = test_df))

# TRAIN metrics
yTrain <- train_df$SalePrice
lrmTrainMAE  <- mae(yTrain, lrmPredTrain)
lrmTrainRMSE <- sqrt(mse(yTrain, lrmPredTrain))
lrmTrainR2   <- summary(model_lm)$r.squared

# TEST metrics
yTest <- test_df$SalePrice
lrmTestMAE  <- mae(yTest, lrmPredTest)
lrmTestRMSE <- sqrt(mse(yTest, lrmPredTest))
SSres <- sum((yTest - lrmPredTest)^2)
SStot <- sum((yTest - mean(yTest))^2)
lrmTestR2 <- 1 - SSres/SStot

# Residual plots
plot(lrmPredTrain, yTrain - lrmPredTrain,
     xlab = "Predicted Train Values", ylab = "Residuals",
     main = "(LR) Residuals vs Predicted (Train)"); abline(h = 0, lty = 2)
plot(lrmPredTest,  yTest  - lrmPredTest,
     xlab = "Predicted Test Values",  ylab = "Residuals",
     main = "(LR) Residuals vs Predicted (Test)");  abline(h = 0, lty = 2)

# Added-variable plots (if 'car' is installed)
if (has_car) {
  avPlots(model_lm, ask = FALSE)
} else {
  message("Tip: install 'car' for avPlots: install.packages('car')")
}

# Print LR results
cat("===============================================\n")
cat("Linear Regression Model, Train Vs Test Results:\n")
cat("-----------------------------------------------\n")
cat(sprintf("(Train) Mean Absolute Error: %.3f\n", lrmTrainMAE))
cat(sprintf("(Test)  Mean Absolute Error: %.3f\n", lrmTestMAE))
cat("-----------------------------------------------\n")
cat(sprintf("(Train) Root Mean Squared Error: %.3f\n", lrmTrainRMSE))
cat(sprintf("(Test)  Root Mean Squared Error: %.3f\n", lrmTestRMSE))
cat("-----------------------------------------------\n")
cat(sprintf("(Train) R-squared: %.3f\n", lrmTrainR2))
cat(sprintf("(Test)  R-squared: %.3f\n", lrmTestR2))

# ==========================
# 7) Decision Tree (DTM)
# ==========================
# Train a "wide" tree and then prune to ≤ 99 splits (approx. MaxNumSplits)
ctrl <- rpart.control(minbucket = 8, cp = 0.0, maxdepth = 30, xval = 10)
tree0 <- rpart(SalePrice ~ ., data = train_df, method = "anova", control = ctrl)

nsplits <- function(tree) sum(tree$frame$var != "<leaf>")

if (nsplits(tree0) > 99) {
  cpTab <- tree0$cptable
  ok <- which(cpTab[, "nsplit"] <= 99)
  if (length(ok) > 0) {
    best_in_limit <- ok[ which.min(cpTab[ok, "xerror"]) ]
    cp_best <- cpTab[best_in_limit, "CP"]
  } else {
    best_in_limit <- which.min(cpTab[, "xerror"])
    cp_best <- cpTab[best_in_limit, "CP"]
  }
  dtm <- prune(tree0, cp = cp_best)
} else {
  dtm <- tree0
}

cat(sprintf("Final splits in the tree: %d\n", nsplits(dtm)))

# Plot tree (adjust cex/tweak/faclen if text doesn't fit)
rpart.plot(dtm, type = 2, extra = 101, fallen.leaves = TRUE,
           main = "Decision Tree (rpart)", cex = 0.75, tweak = 0.95, faclen = 6)

# Predictions
dtmPredTrain <- predict(dtm, newdata = train_df)
dtmPredTest  <- predict(dtm, newdata = test_df)

# TRAIN metrics
dtmTrainMAE  <- mae(yTrain, dtmPredTrain)
dtmTrainRMSE <- sqrt(mse(yTrain, dtmPredTrain))
SSres_tr <- sum((yTrain - dtmPredTrain)^2)
SStot_tr <- sum((yTrain - mean(yTrain))^2)
dtmTrainR2 <- 1 - SSres_tr / SStot_tr

# TEST metrics
dtmTestMAE  <- mae(yTest, dtmPredTest)
dtmTestRMSE <- sqrt(mse(yTest, dtmPredTest))
SSres_te <- sum((yTest - dtmPredTest)^2)
SStot_te <- sum((yTest - mean(yTest))^2)
dtmTestR2 <- 1 - SSres_te / SStot_te

# Residual plots
plot(dtmPredTrain, yTrain - dtmPredTrain,
     xlab = "Predicted Train Values", ylab = "Residuals",
     main = "(DTM) Residuals vs Predicted (Train)"); abline(h = 0, lty = 2)
plot(dtmPredTest,  yTest  - dtmPredTest,
     xlab = "Predicted Test Values",  ylab = "Residuals",
     main = "(DTM) Residuals vs Predicted (Test)");  abline(h = 0, lty = 2)

# Print DTM results
cat("===============================================\n")
cat("===============================================\n")
cat("Decision Tree Model, Train Vs Test Results:\n")
cat("-----------------------------------------------\n")
cat(sprintf("(Train) Mean Absolute Error: %.3f\n", dtmTrainMAE))
cat(sprintf("(Test)  Mean Absolute Error: %.3f\n", dtmTestMAE))
cat("-----------------------------------------------\n")
cat(sprintf("(Train) Root Mean Squared Error: %.3f\n", dtmTrainRMSE))
cat(sprintf("(Test)  Root Mean Squared Error: %.3f\n", dtmTestRMSE))
cat("-----------------------------------------------\n")
cat(sprintf("(Train) R-squared: %.3f\n", dtmTrainR2))
cat(sprintf("(Test)  R-squared: %.3f\n", dtmTestR2))

# ==========================
# 8) LASSO with CV (7-fold)
# ==========================
# Using glmnet, replicating MATLAB's z-score
# (install.packages("glmnet") if missing)
library(glmnet)

# Predictor matrices (without SalePrice)
Xtr_raw <- as.matrix(train_df[, -1, drop = FALSE])
Xte_raw <- as.matrix(test_df[,  -1, drop = FALSE])
ytr <- train_df$SalePrice
yte <- test_df$SalePrice

# Standardize with TRAIN means and sds (like MATLAB's zscore)
mu <- colMeans(Xtr_raw)
sigma <- apply(Xtr_raw, 2, sd)
sigma[sigma == 0 | is.na(sigma)] <- 1  # avoid division by 0

Xtr <- scale(Xtr_raw, center = mu, scale = sigma)
Xte <- scale(Xte_raw, center = mu, scale = sigma)

set.seed(123)
cv <- cv.glmnet(Xtr, ytr, alpha = 1, nfolds = 7, standardize = FALSE)  # FALSE because already z-scored

lambda_opt <- cv$lambda.min
cat("Optimal Lambda:", lambda_opt, "\n")

# Coefficients at optimal lambda (intercept separate)
coef_opt <- coef(cv, s = "lambda.min")
print(coef_opt)

# Final model and predictions
model_lasso <- glmnet(Xtr, ytr, alpha = 1, lambda = lambda_opt, standardize = FALSE)
pred_tr <- as.numeric(predict(model_lasso, newx = Xtr))
pred_te <- as.numeric(predict(model_lasso, newx = Xte))

# Metrics
cat("== Lasso (Train/Test) ==\n")
cat(sprintf("MAE train: %.3f\n", mae(ytr, pred_tr)))
cat(sprintf("RMSE train: %.3f\n", sqrt(mse(ytr, pred_tr))))
cat(sprintf("R2   train: %.3f\n", 1 - sum((ytr - pred_tr)^2) / sum((ytr - mean(ytr))^2)))
cat(sprintf("MAE test : %.3f\n", mae(yte, pred_te)))
cat(sprintf("RMSE test: %.3f\n", sqrt(mse(yte, pred_te))))
cat(sprintf("R2   test: %.3f\n", 1 - sum((yte - pred_te)^2) / sum((yte - mean(yte))^2)))
