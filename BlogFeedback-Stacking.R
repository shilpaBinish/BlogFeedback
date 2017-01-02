### Stacked Generalization ###
library(data.table)
library(foreach)
library(MatrixModels)

library(FNN)
library(glmnet)
library(ranger)
library(xgboost)


# load and combine dataset
train = fread("BlogFeedback-Train.csv")
test = fread("BlogFeedback-Test.csv")

# error measure
mse = function(y_hat, y) {
  mse = mean((y - y_hat)^2)
  
  return(mse)
}

# create design matrices
test_x_sparse = model.Matrix(V281 ~ . - 1, data = test, sparse = T)
train_y = train$V281
test_y = test$V281

# divide training set into k folds
k = 5
cv_index = 1:nrow(train)
cv_index_split = split(cv_index, cut(seq_along(cv_index), k, labels = FALSE))
                 
# meta features from kNN
meta_knn_test = rep(0, nrow(test))
meta_knn_train = foreach(i = 1:k, .combine = c) %do% {
  # split the raining set into two disjoint sets
  train_index = setdiff(1:nrow(train), cv_index_split[[i]])
  train_set1 = model.Matrix(V281 ~ . - 1, data = train[train_index], sparse = T)
  train_set2 = model.Matrix(V281 ~ . - 1, data = train[cv_index_split[[i]]], sparse = T)
  
  # level 0 prediction
  meta_pred = knn.reg(train_set1, train_set2, train[train_index]$V281, k = 22)$pred
  meta_knn_test = meta_knn_test + knn.reg(train_set1, test_x_sparse, train[train_index]$V281, k = 22)$pred / k
  
  return(meta_pred)
}

# meta features from LASSO
meta_glm_test = rep(0, nrow(test))
meta_glm_train = foreach(i = 1:k, .combine = c) %do% {
  # split the raining set into two disjoint sets
  train_index = setdiff(1:nrow(train), cv_index_split[[i]])
  train_set1 = model.Matrix(V281 ~ . - 1, data = train[train_index], sparse = T)
  train_set2 = model.Matrix(V281 ~ . - 1, data = train[cv_index_split[[i]]], sparse = T)
  
  # level 0 prediction
  temp_glm = cv.glmnet(train_set1, train[train_index]$V281, family = "gaussian", alpha = 1)
  meta_pred = predict(temp_glm, newx = train_set2)
  meta_glm_test = meta_glm_test + predict(temp_glm, newx = test_x_sparse) / k
  
  return(meta_pred)
}

# meta features from random forest
meta_rf_test = rep(0, nrow(test))
meta_rf_train = foreach(i = 1:k, .combine = c) %do% {
  # split the raining set into two disjoint sets
  train_index = setdiff(1:nrow(train), cv_index_split[[i]])
  train_set1 = train[train_index]
  train_set2 = train[cv_index_split[[i]]]
  
  # level 0 prediction
  temp_rf = ranger(V281 ~ ., data = train_set1, num.trees = 100, mtry = 80, write.forest = T)
  meta_pred = predict(temp_rf, train_set2)$predictions
  meta_rf_test = meta_rf_test + predict(temp_rf, test)$predictions / k
  
  return(meta_pred)
}

# combine meta features
train_sg = cbind(meta_knn_train, meta_glm_train, meta_rf_train)
test_sg = cbind(meta_knn_test, meta_glm_test, meta_rf_test)

train_sg_xgb = xgb.DMatrix(data = train_sg, label = train_y)
test_sg_xgb = xgb.DMatrix(data = test_sg, label = test_y)

# ensemble with boosting
mdl_xgb = xgboost(data = train_sg_xgb, nround = 800, nthread = 4, max_depth = 5, eta = 0.025, subsample = 0.8, gamma = 6)
pred_xgb = predict(mdl_xgb, test_sg_xgb)
mse(pred_xgb, test_y)