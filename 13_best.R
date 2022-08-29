#### STAT 639 Project R Codes For Best Results Of Classification Problem
#### Group 13: Grace Atama, Charan Jirra, Jiahong Zhou


## Loading Libraries
library(caret)
library(tidyverse)
library(glmnet)
library(e1071)
library(mlbench)
library(caret)
library(MASS)
library(gbm)
library(kernlab)
library(Boruta)

## Loading the file
load("class_data.Rdata")

## Support Vector Machine (SVM) with boruta
Nested_SVM_boruta = function(x, y, xtest, outerK=10){ # the nested cv function
  set.seed(1)
  outerFolds = createFolds(y, k=outerK) # create folds for CV
  outerFoldError = rep(0, outerK) # place holder to save error for each fold
  pred = c()
  for (i in 1:outerK){ # outer loop
    set.seed(i)
    xInner = x[-outerFolds[[i]],] # all data for the inner loop
    yInner = y[-outerFolds[[i]]]
    dat = data.frame(xInner, yInner)
    boruta = Boruta(yInner ~ ., data = dat, doTrace = 0, maxRuns = 70)
    selected_attribute = getSelectedAttributes(boruta)
    cvInner.tune = train(yInner ~ .,data = dat[c(selected_attribute,"yInner")],
                         method = "svmRadialSigma",
                         trControl = trainControl(method = "repeatedcv", 
                                                  number = 10, 
                                                  repeats = 5,
                                                  verboseIter = FALSE),
                         verbose = 0)
    xOuter = x[outerFolds[[i]],] # Outer lop data
    yOuter = y[outerFolds[[i]]]
    datOuter = data.frame(xOuter, yOuter)
    outerFoldPred=predict(cvInner.tune,datOuter[c(selected_attribute)]) # Outer loop prediction
    outerFoldError[i] = mean(outerFoldPred != y[outerFolds[[i]]]) # misclassification
    testPred = predict(cvInner.tune,data.frame(xtest)[c(selected_attribute)]) # prediction on given test set
    pred = cbind(pred, as.numeric(levels(testPred))[testPred])
  }
  # CV error rate is total misclassification rate
  cvError = mean(outerFoldError)
  pred_response = ifelse(apply(pred,1,mean)>0.5, 1, 0) # average prediction
  returnList = list("outerError"=outerFoldError, "cvError"=cvError, "pred"=pred_response, "parameters"=cvInner.tune$bestTune)
  return(returnList) ### return value
}

## SVM with boruta results
xScale = scale(x)
yFac = factor(y)
xTestScale = scale(xnew)


Output.SVM_boruta = Nested_SVM_boruta(xScale, yFac, xTestScale)
y_pred.SVM_boruta = Output.SVM_boruta$pred
cv_error.SVM_boruta = Output.SVM_boruta$cvError
SVM_boruta.parameters = Output.SVM_boruta$parameters

ynew = y_pred.SVM_boruta
test_error = cv_error.SVM_boruta

save(ynew,test_error,file="13.RData")

