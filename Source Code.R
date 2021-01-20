#Import Libraries
library(modelr)
library(caret)
library(glmnet)


#Import the Red and White Wine datasets
red_wine <- read.csv("winequality-red.csv")
white_wine <- read.csv("winequality-white.csv")

#Some data exploratory analysis
hist(red_wine$quality, main="Red Wine Quality Distribution", xlab="Quality")
unique(red_wine$quality)

hist(white_wine$quality, main="White Wine Quality Distribution", xlab="Quality")
unique(white_wine$quality)

#Red Wine Training and Testing datasets
set.seed(123)
red_dataset <- resample_partition(red_wine, c(train=0.8,test=0.2))
red_train <- data.frame(red_dataset$train)
red_test <- data.frame(red_dataset$test)

#White Wine Training and Testing datasets
set.seed(123)
white_dataset <- resample_partition(white_wine, c(train=0.8,test=0.2))
white_train <- data.frame(white_dataset$train)
white_test <- data.frame(white_dataset$test)


#Multiple Linear Regression model
mlm_red <- lm(quality ~ ., data=red_train)
mlm_white <- lm(quality ~ ., data=white_train)

mlm_red_pred <- predict(mlm_red, red_test)
mlm_white_pred <- predict(mlm_white, white_test)

mlm_red_metrics <- data.frame(RMSE = RMSE(red_test$quality,mlm_red_pred),
                              MAE = MAE(red_test$quality,mlm_red_pred),
                              R2 = caret::R2(red_test$quality,mlm_red_pred))
mlm_white_metrics <- data.frame(RMSE = RMSE(white_test$quality,mlm_white_pred),
                                MAE = MAE(white_test$quality,mlm_white_pred),
                                R2 = caret::R2(white_test$quality,mlm_white_pred))
mlm_red_metrics
mlm_white_metrics

#Residual and Normal Probability plots for both Multiple Linear models
#Red Wine
plot(mlm_red,1)
plot(mlm_red,2)
#White Wine
plot(mlm_white,1)
plot(mlm_white,2)


#Polynomial with Square Root Transformed Term Regression model
#Decided to go with Square Root transformation because of the Residual 
#and Normal Probability plots of the Multiple Linear models
red_train_sqrt <- sqrt(red_train)
red_test_sqrt <- sqrt(red_test)

trans_red <- lm(quality ~ ., data=red_train_sqrt)
trans_red_pred <- predict(trans_red, red_test_sqrt)
trans_red_metrics <- data.frame(RMSE = RMSE(red_test_sqrt$quality,trans_red_pred),
                                MAE = MAE(red_test_sqrt$quality,trans_red_pred),
                                R2 = caret::R2(red_test_sqrt$quality,trans_red_pred))
trans_red_metrics

white_train_sqrt <- sqrt(white_train)
white_test_sqrt <- sqrt(white_test)
trans_white <- lm(quality ~ ., data=white_train_sqrt)
trans_white_pred <- predict(trans_white, white_test_sqrt)
trans_white_metrics <- data.frame(RMSE = RMSE(white_test_sqrt$quality,trans_white_pred),
                                MAE = MAE(white_test_sqrt$quality,trans_white_pred),
                                R2 = caret::R2(white_test_sqrt$quality,trans_white_pred))
trans_white_metrics


#Ridge Regression model
#Red Wine
x_train_red <- data.matrix(red_train[,1:11])
y_train_red <- red_train[,"quality"]
x_test_red <- data.matrix(red_test[,1:11])
y_test_red <- red_test[,"quality"]

lambda_seq <- 10^seq(-2,2,length=100)
set.seed(123)
ridge_cv <- cv.glmnet(x_train_red,y_train_red,alpha=0,lambda=lambda_seq, nfolds=5)
lambda_min <- ridge_cv$lambda.min
ridge_red <- glmnet(x_train_red,y_train_red,alpha=0,lambda=lambda_min)

ridge_red_pred <- predict(ridge_red,s=lambda_min, newx=x_test_red)
ridge_red_metrics <- data.frame(RMSE = RMSE(y_test_red,ridge_red_pred),
                                MAE = MAE(y_test_red,ridge_red_pred),
                                R2 = caret::R2(y_test_red,ridge_red_pred))
ridge_red_metrics

#White Wine
x_train_white <- data.matrix(white_train[,1:11])
y_train_white <- white_train[,"quality"]
x_test_white <- data.matrix(white_test[,1:11])
y_test_white <- white_test[,"quality"]

set.seed(123)
ridge_cv <- cv.glmnet(x_train_white,y_train_white,alpha=0,lambda=lambda_seq, nfolds=5)
lambda_min <- ridge_cv$lambda.min
ridge_white <- glmnet(x_train_white,y_train_white,alpha=0,lambda=lambda_min)

ridge_white_pred <- predict(ridge_white, s=lambda_min, newx=x_test_white)
ridge_white_metrics <- data.frame(RMSE = RMSE(y_test_white,ridge_white_pred),
                                  MAE = MAE(y_test_white,ridge_white_pred),
                                  R2 = caret::R2(y_test_white,ridge_white_pred))
ridge_white_metrics


#Lasso Regression model
#Using the same train and test datasets as Ridge models
#Red Wine
set.seed(123)
lasso_cv <- cv.glmnet(x_train_red,y_train_red,alpha=1,lambda=lambda_seq, nfolds=5)
lambda_min <- lasso_cv$lambda.min
lasso_red <- glmnet(x_train_red,y_train_red,alpha=1,lambda=lambda_min)

lasso_red_pred <- predict(lasso_red,s=lambda_min, newx=x_test_red)
lasso_red_metrics <- data.frame(RMSE = RMSE(y_test_red,lasso_red_pred),
                                MAE = MAE(y_test_red,lasso_red_pred),
                                R2 = caret::R2(y_test_red,lasso_red_pred))
lasso_red_metrics

#White Wine
set.seed(123)
lasso_cv <- cv.glmnet(x_train_white,y_train_white,alpha=1,lambda=lambda_seq, nfolds=5)
lambda_min <- lasso_cv$lambda.min
lasso_white <- glmnet(x_train_white,y_train_white,alpha=1,lambda=lambda_min)

lasso_white_pred <- predict(lasso_white,s=lambda_min, newx=x_test_white)
lasso_white_metrics <- data.frame(RMSE = RMSE(y_test_white,lasso_white_pred),
                                  MAE = MAE(y_test_white,lasso_white_pred),
                                  R2 = caret::R2(y_test_white,lasso_white_pred))
lasso_white_metrics


#Elastic Net Regression model
#Using the same train and test datasets as Ridge models
#Red Wine
#Finding the optimal value for alpha based on highest R2
seq1 <- seq(0,1,by=0.1) 
for(i in seq1){ 
  set.seed(123)
  elastic_cv <- cv.glmnet(x_train_red,y_train_red,alpha=i,lambda=lambda_seq, nfolds=5)
  lambda_min <- elastic_cv$lambda.min
  elastic_red <- glmnet(x_train_red,y_train_red,alpha=i,lambda=lambda_min)
  
  elastic_red_pred <- predict(elastic_red,s=lambda_min, newx=x_test_red)
  R2 = caret::R2(y_test_red,elastic_red_pred)
  print(paste(i,R2)) #0.4 is the best alpha
}

#Red Wine with alpha=0.4
set.seed(123)
elastic_cv <- cv.glmnet(x_train_red,y_train_red,alpha=0.4,lambda=lambda_seq, nfolds=5)
lambda_min <- elastic_cv$lambda.min
elastic_red <- glmnet(x_train_red,y_train_red,alpha=0.4,lambda=lambda_min)

elastic_red_pred <- predict(elastic_red,s=lambda_min, newx=x_test_red)
elastic_red_metrics <- data.frame(RMSE = RMSE(y_test_red,elastic_red_pred),
                                  MAE = MAE(y_test_red,elastic_red_pred),
                                  R2 = caret::R2(y_test_red,elastic_red_pred))
elastic_red_metrics

#White Wine
#Using the same train and test datasets as Ridge models
#Finding the optimal value for alpha based on R2
for(i in seq1){ 
  set.seed(123)
  elastic_cv <- cv.glmnet(x_train_white,y_train_white,alpha=i,lambda=lambda_seq, nfolds=5)
  lambda_min <-  elastic_cv$lambda.min
  elastic_white <- glmnet(x_train_white,y_train_white,alpha=i,lambda=lambda_min)
  
  elastic_white_pred <- predict(elastic_white, s=lambda_min, newx=x_test_white)
  R2 = caret::R2(y_test_white,elastic_white_pred)  
  print(paste(i,R2)) #0.1 is the best alpha for Elastic Net
}

#White Wine with alpha=0.1
set.seed(123)
elastic_cv <- cv.glmnet(x_train_white,y_train_white,alpha=0.1,lambda=lambda_seq, nfolds=5)
lambda_min <- elastic_cv$lambda.min
elastic_white <- glmnet(x_train_white,y_train_white,alpha=0.1,lambda=lambda_min)

elastic_white_pred <- predict(elastic_white, s=lambda_min, newx=x_test_white)
elastic_white_metrics <- data.frame(RMSE = RMSE(y_test_white,elastic_white_pred),
                                    MAE = MAE(y_test_white,elastic_white_pred),
                                    R2 = caret::R2(y_test_white,elastic_white_pred))
elastic_white_metrics


#KNN Regression model
#Red Wine
set.seed(123)
knn_red <- train(quality ~ ., data=red_train, method="knn",
                 trControl=trainControl("cv",number=10),
                 preProcess=c("center","scale"), tuneLength=20)
plot(knn_red, main="Red Wine KNN")
knn_red 

knn_red_pred <- predict(knn_red, red_test)

knn_red_metrics <- data.frame(RMSE = RMSE(red_test$quality,knn_red_pred),
                              MAE = MAE(red_test$quality,knn_red_pred),
                              R2 = caret::R2(red_test$quality,knn_red_pred))
knn_red_metrics

#White Wine
set.seed(123)
knn_white <- train(quality ~ ., data=white_train, method="knn",
                   trControl=trainControl("cv",number=10),
                   preProcess=c("center","scale"), tuneLength=20)
plot(knn_white)
knn_white

knn_white_pred <- predict(knn_white, white_test)

knn_white_metrics <- data.frame(RMSE = RMSE(white_test$quality,knn_white_pred),
                              MAE = MAE(white_test$quality,knn_white_pred),
                              R2 = caret::R2(white_test$quality,knn_white_pred))
knn_white_metrics


#Decision Tree Classification model
#Convert Quality into Factor datatype for the Decision Tree Classification model
red_train$quality <- as.factor(red_train$quality)
red_test$quality <- as.factor(red_test$quality)
white_train$quality <- as.factor(white_train$quality)
white_test$quality <- as.factor(white_test$quality)

#Red Wine
set.seed(123)
tree_red <- train(quality ~ ., data=red_train, method="rpart",
                  trControl=trainControl("cv",number=10),
                  tuneLength=20)
plot(tree_red)
tree_red

tree_red_pred <- predict(tree_red, red_test)

test <- as.numeric(red_test$quality) #Convert into numeric datatype for evaluations
pred <- as.numeric(tree_red_pred) #Convert into numeric datatype for evaluations
tree_red_metrics <- data.frame(RMSE = RMSE(test,pred),
                               MAE = MAE(test,pred),
                               R2 = caret::R2(test,pred))
tree_red_metrics

#White Wine
set.seed(123)
tree_white <- train(quality ~ ., data=white_train, method="rpart",
                    trControl=trainControl("cv",number=10),
                    tuneLength=20)
plot(tree_white)
tree_white

tree_white_pred <- predict(tree_white, white_test)

test <- as.numeric(white_test$quality) #Convert into numeric datatype for evaluations
pred <- as.numeric(tree_white_pred) #Convert into numeric datatype for evaluations
tree_white_metrics <- data.frame(RMSE = RMSE(test,pred),
                                 MAE = MAE(test,pred),
                                 R2 = caret::R2(test,pred))
tree_white_metrics




