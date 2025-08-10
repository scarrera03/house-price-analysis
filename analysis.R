# ==========================
# House Price Analysis - Ames
# ==========================

# Paquetes (instalalos 1 sola vez fuera del script si faltan)
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(Metrics)
# Added-variable plots (opcional)
has_car <- requireNamespace("car", quietly = TRUE)
if (has_car) library(car)

# --- 1) Cargar dataset ---
houses <- read.csv("/Users/silvinaguadalupecarrerascholz/house-price-analysis/houses.csv")
stopifnot("SalePrice" %in% names(houses))

# --- 2) Holdout 70/30 ---
set.seed(123)
idx <- createDataPartition(houses$SalePrice, p = 0.7, list = FALSE)
train <- houses[idx, ]
test  <- houses[-idx, ]

# --- 3) Remover IDs que no aportan (usa solo las que existan) ---
train <- train %>% select(-any_of(c("Id","Order","PID")))
test  <- test  %>% select(-any_of(c("Id","Order","PID")))

# --- 4) Imputación y tipado coherente (como en MATLAB) ---
# Asegurar que SalePrice no tenga NA
train <- train[!is.na(train$SalePrice), ]
test  <- test[!is.na(test$SalePrice), ]

num_cols  <- names(which(sapply(train, is.numeric)))
char_cols <- names(which(sapply(train, is.character)))
fact_cols <- names(which(sapply(train, is.factor)))

# a) Numéricos → mediana (de train)
for (nm in num_cols) {
  med <- suppressWarnings(median(train[[nm]], na.rm = TRUE))
  if (is.finite(med)) {
    train[[nm]][is.na(train[[nm]])] <- med
    if (nm %in% names(test)) test[[nm]][is.na(test[[nm]])] <- med
  } else {
    # Columna totalmente NA: se elimina en ambos
    train[[nm]] <- NULL
    if (nm %in% names(test)) test[[nm]] <- NULL
  }
}

# b) Categóricas → "Unknown" y niveles fijos de train
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

# --- 5) One-hot encoding (equivalente a onehotencode de MATLAB) ---
dmy <- dummyVars(SalePrice ~ ., data = train, fullRank = TRUE, na.action = na.pass)
Xtr <- data.frame(predict(dmy, newdata = train))
Xte <- data.frame(predict(dmy, newdata = test))

# Alinear columnas entre train y test (mismas features y orden)
missing_in_test <- setdiff(names(Xtr), names(Xte))
if (length(missing_in_test) > 0) Xte[missing_in_test] <- 0
extra_in_test <- setdiff(names(Xte), names(Xtr))
if (length(extra_in_test) > 0) Xte <- Xte[, setdiff(names(Xte), extra_in_test), drop = FALSE]
Xte <- Xte[, names(Xtr), drop = FALSE]

train_df <- cbind(SalePrice = train$SalePrice, Xtr)
test_df  <- cbind(SalePrice = test$SalePrice,  Xte)

# (Opcional) quitar near-zero variance
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

# Predicciones
lrmPredTrain <- as.vector(predict(model_lm, newdata = train_df))
lrmPredTest  <- as.vector(predict(model_lm,  newdata = test_df))

# Métricas TRAIN
yTrain <- train_df$SalePrice
lrmTrainMAE  <- mae(yTrain, lrmPredTrain)
lrmTrainRMSE <- sqrt(mse(yTrain, lrmPredTrain))
lrmTrainR2   <- summary(model_lm)$r.squared

# Métricas TEST
yTest <- test_df$SalePrice
lrmTestMAE  <- mae(yTest, lrmPredTest)
lrmTestRMSE <- sqrt(mse(yTest, lrmPredTest))
SSres <- sum((yTest - lrmPredTest)^2)
SStot <- sum((yTest - mean(yTest))^2)
lrmTestR2 <- 1 - SSres/SStot

# Gráficos de residuos
plot(lrmPredTrain, yTrain - lrmPredTrain,
     xlab = "Predicted Train Values", ylab = "Residuals",
     main = "(LR) Residuals vs Predicted (Train)"); abline(h = 0, lty = 2)
plot(lrmPredTest,  yTest  - lrmPredTest,
     xlab = "Predicted Test Values",  ylab = "Residuals",
     main = "(LR) Residuals vs Predicted (Test)");  abline(h = 0, lty = 2)

# Added-variable plots (si tenés 'car' instalado)
if (has_car) {
  avPlots(model_lm, ask = FALSE)
} else {
  message("Tip: instalá 'car' para avPlots: install.packages('car')")
}

# Print resultados LR
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
# Entrenar árbol "amplio" y luego podar para ≤ 99 splits (aprox. MaxNumSplits)
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

cat(sprintf("Splits finales del árbol: %d\n", nsplits(dtm)))

# Graficar árbol (ajustá cex/tweak/faclen si el texto no entra)
rpart.plot(dtm, type = 2, extra = 101, fallen.leaves = TRUE,
           main = "Decision Tree (rpart)", cex = 0.75, tweak = 0.95, faclen = 6)

# Predicciones
dtmPredTrain <- predict(dtm, newdata = train_df)
dtmPredTest  <- predict(dtm, newdata = test_df)

# Métricas TRAIN
dtmTrainMAE  <- mae(yTrain, dtmPredTrain)
dtmTrainRMSE <- sqrt(mse(yTrain, dtmPredTrain))
SSres_tr <- sum((yTrain - dtmPredTrain)^2)
SStot_tr <- sum((yTrain - mean(yTrain))^2)
dtmTrainR2 <- 1 - SSres_tr / SStot_tr

# Métricas TEST
dtmTestMAE  <- mae(yTest, dtmPredTest)
dtmTestRMSE <- sqrt(mse(yTest, dtmPredTest))
SSres_te <- sum((yTest - dtmPredTest)^2)
SStot_te <- sum((yTest - mean(yTest))^2)
dtmTestR2 <- 1 - SSres_te / SStot_te

# Gráficos de residuos
plot(dtmPredTrain, yTrain - dtmPredTrain,
     xlab = "Predicted Train Values", ylab = "Residuals",
     main = "(DTM) Residuals vs Predicted (Train)"); abline(h = 0, lty = 2)
plot(dtmPredTest,  yTest  - dtmPredTest,
     xlab = "Predicted Test Values",  ylab = "Residuals",
     main = "(DTM) Residuals vs Predicted (Test)");  abline(h = 0, lty = 2)

# Print resultados DTM
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


# LASSO con CV (7-fold) usando glmnet, replicando z-score de MATLAB

install.packages("glmnet")
library(glmnet)
library(Metrics)

# Matrices de predictores (sin SalePrice)
Xtr_raw <- as.matrix(train_df[, -1, drop = FALSE])
Xte_raw <- as.matrix(test_df[,  -1, drop = FALSE])
ytr <- train_df$SalePrice
yte <- test_df$SalePrice

# Estandarizar con medias y sds de TRAIN (como zscore de MATLAB)
mu <- colMeans(Xtr_raw)
sigma <- apply(Xtr_raw, 2, sd)
sigma[sigma == 0 | is.na(sigma)] <- 1  # evitar divisiones por 0

Xtr <- scale(Xtr_raw, center = mu, scale = sigma)
Xte <- scale(Xte_raw, center = mu, scale = sigma)

set.seed(123)
cv <- cv.glmnet(Xtr, ytr, alpha = 1, nfolds = 7, standardize = FALSE)  # FALSE porque ya z-scoreamos

lambda_opt <- cv$lambda.min
cat("Optimal Lambda:", lambda_opt, "\n")

# Coeficientes en lambda óptimo (incluye intercepto aparte)
coef_opt <- coef(cv, s = "lambda.min")
print(coef_opt)

# Modelo final y predicciones
model_lasso <- glmnet(Xtr, ytr, alpha = 1, lambda = lambda_opt, standardize = FALSE)
pred_tr <- as.numeric(predict(model_lasso, newx = Xtr))
pred_te <- as.numeric(predict(model_lasso, newx = Xte))

# Métricas
cat("== Lasso (Train/Test) ==\n")
cat(sprintf("MAE train: %.3f\n", mae(ytr, pred_tr)))
cat(sprintf("RMSE train: %.3f\n", sqrt(mse(ytr, pred_tr))))
cat(sprintf("R2   train: %.3f\n", 1 - sum((ytr - pred_tr)^2) / sum((ytr - mean(ytr))^2)))
cat(sprintf("MAE test : %.3f\n", mae(yte, pred_te)))
cat(sprintf("RMSE test: %.3f\n", sqrt(mse(yte, pred_te))))
cat(sprintf("R2   test: %.3f\n", 1 - sum((yte - pred_te)^2) / sum((yte - mean(yte))^2)))

