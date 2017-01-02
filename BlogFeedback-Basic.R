### Simple Models ###
library(data.table)
library(MatrixModels)

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
train_x = model.Matrix(V281 ~ . - 1, data = train, sparse = F)
train_y = train$V281

test_x = model.Matrix(V281 ~ . - 1, data = test, sparse = F)
test_y = test$V281

train_xgb = xgb.DMatrix(data = as.matrix(train_x), label = train_y)
test_xgb = xgb.DMatrix(data = as.matrix(test_x), label = test_y)

# try LASSO
mdl_lasso = cv.glmnet(train_x, train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = test_x)
mse(pred_lasso, test_y)

# try random forest
mdl_rf = ranger(V281 ~ ., data = train, num.trees = 500, mtry = 120, write.forest = T)
pred_rf = predict(mdl_rf, test)
mse(pred_rf$predictions, test_y)

# try XGboost
mdl_xgb = xgboost(data = train_xgb, nround = 750, nthread = 4, max_depth = 6, eta = 0.025, subsample = 0.7, gamma = 3)
pred_xgb = predict(mdl_xgb, test_xgb)
mse(pred_xgb, test_y)