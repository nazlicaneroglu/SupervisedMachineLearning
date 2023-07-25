### file path etc
#rm(list=ls())
options(scipen=6, digits=4)
##packages anf libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, tidyverse, tinytex, rmarkdown, glmnet, matlib, MASS,pdist, dplyr,plotrix, kernlab,ranger, randomForest, )
library(matlib)
library(glmnet, quietly = TRUE)
library(caTools)

##Data
music<-read.csv("/Users/ekuleru/Desktop/TI Second year/Supervised Machine Learning/Final Assignment/udemy course dataset/Udemymusic.csv")
df<- subset(music, select = -c(url, course_id, course_title, published_timestamp, subject) )
music<-df[,c(2,1,3,4,5,6,7)]
#Check whether we have omitted variables
sum(complete.cases(music))
summary(music)
#Organising and scaling
y<- as.vector(music$num_subscriber)
X <- model.matrix(num_subscribers ~ . , data = music)
X[, 2:9] <- scale(X[, 2:9])
X <- X[, -1]
set.seed(8913)

##Ridge Regression
result_ridge <- glmnet(X, y, alpha = 0, lambda = 10^seq(-2, 6, length.out = 50),
                 standardize = FALSE)
plot(result_ridge, xvar = "lambda", label = TRUE, las = 1)
legend("topright", lwd = 1, col = 1:6, bg = "white", legend = pasteCols(t(cbind(1:ncol(X), " ",colnames(X)))), cex = .5)
result_ridge.cv<- cv.glmnet(X, y, alpha = 0, lambda = 10^seq(-2, 6, length.out = 50),
                       standardize = FALSE)
print(result_ridge.cv$lambda.min)# Best cross validated lambda
##Min lambda is 790.6

## To plot Root Mean Squared Error (RMSE) to be on the same scale as y:
result_ridge.cv$cvm <- result_ridge.cv$cvm^0.5
result_ridge.cv$cvup <- result_ridge.cv$cvup^0.5
result_ridge.cv$cvlo <- result_ridge.cv$cvlo^0.5
plot(result_ridge.cv, ylab = "Root Mean-Squared Error")
#lowest RMSE= 4164
#Results with the best lambda
finalresult_ridge <- glmnet(X, y, alpha = 0, lambda = result_ridge.cv$lambda.min,
                 intercept = TRUE)
print(finalresult_ridge$beta)

#however, the umber of parameters is very small compared to the number of observations.
#that's why, i also tried t replace each predictor by its polynomial basis and model interaction

deg<-7
X.poly.ridge <- model.matrix(~ 0 + poly(X[, 1], degree = deg) 
                       + poly(X[, 2], degree = deg) 
                       + poly(X[, 3], degree = deg)
                       + poly(X[, 7], degree = deg)
                       + poly(X[, 8], degree = deg), data = as.data.frame(X[, 2:8]))
X.inter.action <- model.matrix( ~ .^2, data = as.data.frame(X.poly.ridge))
dim(X.inter.action)
#I dont know why i could not use 4th,5th and 6th column. I guess that since they are dummy variables there
# don't have enough unique points and it creates problem when we try to create higher degree polynomial

result_ridgepoly.cv <- cv.glmnet(X.inter.action, y, alpha = 0,
                       lambda = 10^seq(-2, 10, length.out = 50), nfolds = 10)
print(result_ridgepoly.cv$lambda.min) # Best cross validated lambda
##Min lambda is 1389

print(result_ridgepoly.cv$lambda.1se) # Conservative est. of best lambda (1 stdev)

#RSME for ridge polynomial
plot(result_ridgepoly.cv$lambda, result_ridgepoly.cv$cvm^.5, log = "x", col = "red", type = "p", pch = 20,
        xlab = expression(lambda), ylab = "RMSE", las = 1)
#another RSME graph 
result_ridgepoly.cv$cvm <- result_ridgepoly.cv$cvm^0.5
result_ridgepoly.cv$cvup <- result_ridgepoly.cv$cvup^0.5
result_ridgepoly.cv$cvlo <- result_ridgepoly.cv$cvlo^0.5
plot(result_ridgepoly.cv, ylab = "Root Mean-Squared Error")
#lowest RMSE we get is = 5445
#Results with the best lambda
finalresult_ridgepoly <- glmnet(X, y, alpha = 0, lambda = result_ridgepoly.cv$lambda.min,
                            intercept = TRUE)
print(finalresult_ridgepoly$beta)

## Lasso Regression

result_lasso <- glmnet(X, y, alpha = 1, lambda = 10^seq(-2, 6, length.out = 50),
                       standardize = FALSE)
plot(result_lasso, xvar = "lambda", label = TRUE, las = 1)
legend("topright", lwd = 1, col = 1:6, bg = "white", legend = pasteCols(t(cbind(1:ncol(X), " ",colnames(X)))), cex = .6)
result_lasso.cv<- cv.glmnet(X, y, alpha = 1, lambda = 10^seq(-2, 6, length.out = 50),
                            standardize = FALSE)
#RSME for Lasso = 4068

result_lasso.cv$cvm <- result_lasso.cv$cvm^0.5
result_lasso.cv$cvup <- result_lasso.cv$cvup^0.5
result_lasso.cv$cvlo <- result_lasso.cv$cvlo^0.5
plot(result_lasso.cv, ylab = "Root Mean-Squared Error")

##Lasso does not want to penalize much meaning that there are not many high influence features

print(result_lasso.cv$lambda.min)# Best cross validated lambda
##Min lambda is 175.8
#Results with the best lambda
finalresult.lasso <- glmnet(X, y, alpha = 1,
                            lambda = result_lasso.cv$lambda.min)
print(finalresult.lasso$beta)

#use same lambda for comparison
finalresult.lasso_lambda <- glmnet(X, y, alpha = 1,
                            lambda = result_ridgepoly.cv$lambda.min)
print(finalresult.lasso_lambda$beta)

finalresult.lasso_lambda2 <- glmnet(X, y, alpha = 1,
                                   lambda = result_ridge.cv$lambda.min)
print(finalresult.lasso_lambda2$beta)

ridge_lassolambda<-glmnet(X, y, alpha = 0, lambda = result_lasso.cv$lambda.min,
          standardize = FALSE)
print(ridge_lassolambda$beta)
#We dont see a difference when we switch lambdas


## Random Forest
#Organizing the data
bag_data<-as.data.frame(X[, 2:8])
sort(unique(colnames(bag_data)))
names(bag_data) <- make.names(names(bag_data))

#Bagging implementation
bagging_music <- randomForest(y ~ ., data = bag_data, mtry = 7, ntree = 500, importance = TRUE, do.trace = TRUE)
bagging_music
sqrt(bagging_music$mse[length(bagging_music$mse)])
#RMSE=4159

#Random forest implementation
plot(bagging_music, lwd = 2)
varImpPlot(bagging_music)
# it seems that number of reviews is the most important indicator. we do not know the causality though
rf_music <- randomForest(y ~ ., data = bag_data, mtry = 4, ntree = 500, importance = TRUE, do.trace = TRUE) #look at 4 features
rf_music
plot(rf_music, lwd = 2)
varImpPlot(rf_music)
randomForest::importance(rf_music)
sqrt(rf_music$mse[length(rf_music$mse)])
#RMSE=4114

rf_music2 <- randomForest(y ~ ., data = bag_data, mtry = 2, ntree = 500, importance = TRUE, do.trace = TRUE)#look at 2 features
rf_music2
plot(rf_music2 , lwd = 2)
varImpPlot(rf_music2 )
sqrt(rf_music2 $mse[length(rf_music2 $mse)])
#RMSE=4150

#Also apply permutation just for the curiosity
music_bag_ranger <- ranger(y ~ ., data = bag_data, mtry = 4,
                           num.trees = 500, importance = "permutation")
music_bag_ranger

#higher r^2 in bagging-->intuition: no overfitting as we increase number of features bc only few are significant

#Prediction (something went wrong here)

#Create training and test set before standardization
split = sample.split(music$num_subscribers, SplitRatio = 0.75)
training_set = subset(music, split == TRUE)
test_set = subset(music, split == FALSE)
#standardize for comparison with previous sections
y<- as.vector(training_set$num_subscriber)
X_train <- model.matrix(num_subscribers ~ . , data = training_set)
X_train[, 2:9] <- scale(X_train[, 2:9])
X_train <- X_train[, -1]

X_test <- model.matrix(num_subscribers ~ . , data = test_set)
X_test[, 2:9] <- scale(X_test[, 2:9])
X_test <- X_test[, -1]

bag_testdata<-as.data.frame(X_test[, 2:8])
sort(unique(colnames(bag_testdata)))
names(bag_testdata) <- make.names(names(bag_testdata))

y_pred = predict(bagging_music, newdata = bag_testdata)
testing<- as.numeric(scale(test_set[, 1]))

sqrt(mean((testing - y_pred)^2)) #Something went wrong

cm = table(test_set[, 1], y_pred)
cm
#I guess it is not meaningful to construct this table for non-dummy outcome variables

