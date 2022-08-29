#### STAT 639 Project Related R Codes
#### Group 13: Grace Atama, Charan Jirra, Jiahong Zhou


## Supervised Learning Task - Classification
load("class_data.Rdata")
## First approach: use methods that deal with high-dimensional data
## 1.1 Penalized logistic regression (cv error 0.353)
library(glmnet)
library(caret)
nested_cv_pLog = function(x, y, xtest, outerK=10, innerK=10){ # the nested cv function
  set.seed(1)
  outerFolds = createFolds(y, k=outerK) # create folds for CV
  outerFoldError = rep(0, outerK) # place holder to save error for each fold
  pred = c()
  for (i in 1:outerK){ # outer loop
    xInner = x[-outerFolds[[i]],] # all data for the inner loop
    yInner = y[-outerFolds[[i]]]
    set.seed(i)
    cvInner = cv.glmnet(as.matrix(xInner), yInner, alpha=1, lambda=seq(0,1,0.001),
                        nfolds=innerK, family="binomial", type.measure="class") # inner cv to tune lambda
    outerFoldPred = predict(cvInner, newx=as.matrix(x[outerFolds[[i]],]), 
                            s="lambda.min",type="class") # make predictions
    outerFoldError[i] = mean(outerFoldPred != y[outerFolds[[i]]]) # misclassification
    testPred = predict(cvInner, newx=xtest, s="lambda.min") # predict on given test set
    pred = cbind(pred, testPred)
  }
  ### CV error rate is total misclassification rate
  cvError = mean(outerFoldError)
  finalpred = ifelse(apply(pred, 1, mean)>0.5, 1, 0) # average prediction
  returnList = list("outerError"=outerFoldError, "cvError"=cvError, "pred"=finalpred)
  return(returnList) ### return value
}
xScale = scale(x)
yFac = factor(y)
xTestScale = scale(xnew)
outPLog = nested_cv_pLog(xScale, yFac, xTestScale)

## 1.2 Penalized LDA (cv error 0.315)
library(penalizedLDA)
nested_cv_pLDA = function(x, y, xtest, outerK=10, innerK=10){
  set.seed(1)
  outerFolds = createFolds(y, k=outerK) # create folds for CV
  outerFoldError = rep(0, outerK) # place holder to save error for each fold
  pred = c()
  for (i in 1:outerK){ # outer loop
    xInner = x[-outerFolds[[i]],] # all data for the inner loop
    yInner = y[-outerFolds[[i]]]
    set.seed(i)
    cvInner = PenalizedLDA.cv(xInner, yInner,lambdas=seq(0,1,by=0.01), nfold=innerK) # inner cv to tune lambda
    out = PenalizedLDA(xInner, yInner, xte=x[outerFolds[[i]],],
                       lambda=cvInner$bestlambda, K=cvInner$bestK) # make predictions
    outerFoldError[i] = mean(out$ypred != y[outerFolds[[i]]]) # misclassification
    testPred = unlist(predict(out, xte=xtest)) # predict on test set
    pred = cbind(pred, testPred)
  }
  ### CV error rate is total misclassification rate
  cvError = mean(outerFoldError)
  finalpred = ifelse(apply(pred-1, 1, mean)>0.5, 1, 0) # average prediction
  returnList = list("cvError"=cvError, "pred"=pred)
  return(returnList) ### return value
}
yRecode = y+1 # for penalized LDA the y is required to be in the form of 1, 2, ...
outPLDA = nested_cv_pLDA(x, yRecode, xnew)

## 1.3 Random Forest (OOB error 0.305)  
library(randomForest)
set.seed(1)
outRF = randomForest(x=x, y=yFac)

## 1.4 Pruning (error 0.30)
library(tree)
data = cbind(x, as.factor(y))
set.seed(1)
train = sample(1:nrow(data), 300)
tree.train = tree(y~., data, subset=train)
plot(tree.train)
text(tree.train, pretty=0)
set.seed(1) 
cv.train = cv.tree(tree.train, FUN=prune.misclass)
plot(cv.train)
prune.train = prune.misclass(tree.train, best=14)
plot(prune.train)
text(prune.train, pretty=0)
tree.pred2 = predict(prune.train, data[-train,], type="class")
table(tree.pred2, data[-train,]$y)

## 1.5 Bagging (error 0.28)
rf.train2 = randomForest(y~.,data, mtry=500,ntree=1000) # m=p 
rf.train2
rf.train0 = randomForest(y~.,data, mtry=22,ntree=1000) # m=sprt(p) (error 0.315)
rf.train0
rf.train1 = randomForest(y~.,data, mtry=250,ntree=1000) # m=p/2 (error 0.28)
rf.train1

## 1.6 Naive Bayes (cv error 0.35)
library(naivebayes)
cv_nb = function(x, y, K=10){
  set.seed(1)
  folds = createFolds(y, k=K) # create folds for CV
  foldError = rep(0, K) # place holder to save error for each fold
  for (i in 1:K){ # cv loop
    xTrain = x[-folds[[i]],]
    yTrain = y[-folds[[i]]]
    nbFit = naive_bayes(xTrain, yTrain)
    xTest = data.frame(x[folds[[i]],])
    foldPred = predict(nbFit, xTest, type="class") # make predictions
    foldError[i] = mean(foldPred != y[folds[[i]]]) # misclassification
  }
  ### CV error rate is total misclassification rate
  cvError = mean(foldError)
  return(cvError) ### return value
}
outNB = cv_nb(x, yFac)


## Second approach: feature selection followed by classification algorithms
## 2.1 PCA then LDA (cv error 0.338)
pca = prcomp(x, center=T, scale=T)
pcVar = pca$sdev^2
propVar = pcVar/sum(pcVar)
plot(cumsum(propVar), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained", type = "b")
numPC = which(cumsum(propVar)>0.8)[1] # pcs that explain 80% of the variance

cv_lda = function(x, y, K=10){
  set.seed(1)
  folds = createFolds(y, k=K) # create folds for CV
  foldError = rep(0, K) # place holder to save error for each fold
  for (i in 1:K){ # cv loop
    xTrain = x[-folds[[i]],]
    yTrain = y[-folds[[i]]]
    data = data.frame(yTrain, xTrain)
    ldaFit = lda(yTrain~., data=data)
    xTest = data.frame(x[folds[[i]],])
    foldPred = predict(ldaFit, newdata=xTest)$class # make predictions
    foldError[i] = mean(foldPred != y[folds[[i]]]) # misclassification
  }
  ### CV error rate is total misclassification rate
  cvError = mean(foldError)
  return(cvError) ### return value
}
xPCA = pca$x[,1:numPC]
outLDA = cv_lda(xPCA, yFac)

## 2.2 two sample t-test to select 30 features then logistic regression (cv error 0.32)
# two sample t-test for feature selection
xzero = x[y==0,]
xone = x[y==1,]
tstat = c()
for (i in 1:500) {
  a = t.test(xzero[,i], xone[,i])
  tstat[i] = a$statistic
}
tstat = abs(tstat)
sigVars = order(tstat, decreasing=T) # order the variables by absolute value of t statistic

cv_logistic_t = function(x, y, K=10){ # cv for logistic regression
  set.seed(1)
  folds = createFolds(y, k=K) # create folds for CV
  foldError = rep(0, K) # place holder to save error for each fold
  for (i in 1:K){ # cv loop
    xTrain = x[-folds[[i]],]
    yTrain = y[-folds[[i]]]
    data = data.frame(yTrain, xTrain)
    logisticFit = glm(yTrain~., family=binomial, data=data)
    xTest = data.frame(x[folds[[i]],])
    foldProb = predict(logisticFit, newdata=xTest, type="response") # make predictions
    foldPred = ifelse(foldProb>0.5, 1, 0)
    foldError[i] = mean(foldPred != y[folds[[i]]]) # misclassification
  }
  ### CV error rate is total misclassification rate
  cvError = mean(foldError)
  return(cvError) ### return value
}
x30 = x[,sigVars[1:30]] # 30 features with highest absolute t statistic
outLogistic = cv_logistic_t(x30, yFac)

## 2.3 two sample t-test to select 30 features then LDA (cv error 0.315)
library(MASS)
cv_lda_t = function(x, y, K=10){ # cv for LDA
  set.seed(1)
  folds = createFolds(y, k=K) # create folds for CV
  foldError = rep(0, K) # place holder to save error for each fold
  for (i in 1:K){ # cv loop
    xTrain = x[-folds[[i]],]
    yTrain = y[-folds[[i]]]
    data = data.frame(yTrain, xTrain)
    ldaFit = lda(yTrain~., data=data)
    xTest = data.frame(x[folds[[i]],])
    foldPred = predict(ldaFit, newdata=xTest)$class # make predictions
    #foldPred = ifelse(foldProb>0.5, 1, 0)
    foldError[i] = mean(foldPred != y[folds[[i]]]) # misclassification
  }
  ### CV error rate is total misclassification rate
  cvError = mean(foldError)
  return(cvError) ### return value
}
outLDA = cv_lda_t(x30, yFac)

## 2.4 two sample t-test to select 30 features then RF
new.data = cbind(x30, as.factor(y))
rf.data1 = randomForest(y~., data=new.data, ntree=1000) # error 0.3025
rf.data1
rf.data2 = randomForest(y~., data=new.data, mtry=15, ntree=1000) # mtry = p/2, error 0.2975
rf.data2
rf.data3=randomForest(y~.,data=new.data,mtry=30,ntree=1000) # error 0.31
rf.data3

## 2.5 two sample t-test then penalized logistic regression (cv error 0.308)
x30Scale = scale(x30)
x30Test = select(xnew, all_of(sigVars[1:30]))
x30TestScale = scale(x30Test)
out30PLog = nested_cv_pLog(x30Scale, yFac, x30TestScale)

## 2.6 two sample t-test then penalized LDA (cv error 0.313)
out30PLDA = nested_cv_pLDA(x30, yRecode, x30Test)

## 2.7 two sample t-test then naive bayes (cv error 0.308)
out30NB = cv_nb(x30, yFac)

## 2.8 stepwise logistic regression then penalized logistic regression (cv error 0.163)
interceptOnly = glm(y~1, data=data, family=binomial) # intercept only model
all = glm(y~., data=data, family=binomial) # all predictors model
# perform forward stepwise regression (this line of code takes a few minutes to run)
forward = step(interceptOnly, direction='forward', scope=formula(all), trace=0)
stepVars = names(forward$coefficients)[-1]
stepX = select(x, all_of(stepVars))
stepXScale = scale(stepX)
testX = select(xnew, all_of(stepVars))
xTestScale = scale(testX)
outStepPLog = nested_cv_pLog(stepXScale, yFac, xTestScale)

## 2.9 SVM followed by Boruta feature selection (cv error 0.16) 
library(tidyverse)
library(e1071)
library(mlbench)
library(gbm)
library(kernlab)
library(Boruta)
# Support Vector Machine (SVM) with boruta
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
    xOuter = x[outerFolds[[i]],] # Outer loop data
    yOuter = y[outerFolds[[i]]]
    datOuter = data.frame(xOuter, yOuter)
    outerFoldPred=predict(cvInner.tune,datOuter[c(selected_attribute)]) # Outer loop prediction
    outerFoldError[i] = mean(outerFoldPred != y[outerFolds[[i]]]) # misclassification
    testPred = predict(cvInner.tune,data.frame(xtest)[c(selected_attribute)]) # prediction on given test set
    pred = cbind(pred, as.numeric(levels(testPred))[testPred])
  }
  ### CV error rate is total misclassification rate
  cvError = mean(outerFoldError)
  pred_response = ifelse(apply(pred,1,mean)>0.5, 1, 0) # average prediction
  returnList = list("outerError"=outerFoldError, "cvError"=cvError, "pred"=pred_response, "parameters" = cvInner.tune$bestTune)
  return(returnList) ### return value
}
Output.SVM_boruta = Nested_SVM_boruta(xScale, yFac, xTestScale)

## 2.10 Boosting followed by Boruta feature selection (cv error 0.24) 
Nested_gbm_boruta = function(x, y, xtest, outerK=10){ # the nested cv function
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
    cvInner.tune <- caret::train(yInner ~ .,data = dat[c(selected_attribute,"yInner")],
                                 method = "gbm",
                                 trControl = trainControl(method = "repeatedcv", 
                                                          number = 10, 
                                                          repeats = 5,
                                                          verboseIter = FALSE),
                                 verbose = 0)
    xOuter = x[outerFolds[[i]],] # Outer loop data
    yOuter = y[outerFolds[[i]]]
    datOuter = data.frame(xOuter, yIn = yOuter)
    outerFoldPred=predict(cvInner.tune,datOuter[selected_attribute]) # Outer loop prediction
    outerFoldError[i] = mean(outerFoldPred != y[outerFolds[[i]]]) # misclassification
    testPred = predict(cvInner.tune,data.frame(xtest)[selected_attribute]) # prediction on given test set
    pred = cbind(pred, as.numeric(levels(testPred))[testPred])
  }
  ### CV error rate is total misclassification rate
  cvError = mean(outerFoldError)
  pred_response = ifelse(apply(pred,1,mean)>0.5, 1, 0) # average prediction
  returnList = list("outerError"=outerFoldError, "cvError"=cvError, "pred"=pred_response)
  return(returnList) ### return value
  
}
# Gradient boosting (gbm) with boruta results
Output.gbm_boruta = Nested_gbm_boruta(xScale, yFac, xTestScale)





## Unsupervised Learning Task - Clustering
load("cluster_data.Rdata")
# perform pca to reduce dimension first
pca = prcomp(y, center=T, scale=T)
pcVar = pca$sdev^2
propVar = pcVar/sum(pcVar)
plot(cumsum(propVar), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained", type = "b")
numPC = which(cumsum(propVar)>0.8)[1] # pcs that explain 80% of total variance
df = pca$x[,1:numPC]

library(factoextra)
library(NbClust)
library(cluster)
############ k-means ############
# Elbow method
fviz_nbclust(df, kmeans, nstart=25, method="wss") +
  labs(subtitle="Elbow method - kmeans")
# Silhouette method (9)
fviz_nbclust(df, kmeans, nstart=25, method="silhouette") +
  labs(subtitle="Silhouette method - kmeans")
# Gap statistic (10)
set.seed(111)
fviz_nbclust(df, kmeans, nstart=25,  method="gap_stat") +
  labs(subtitle = "Gap statistic method - kmeans")

############ PAM ############
# Elbow method
fviz_nbclust(df, cluster::pam, method="wss") +
  labs(subtitle="Elbow method - PAM")
# Silhouette method (2)
fviz_nbclust(df, cluster::pam, method="silhouette") +
  labs(subtitle="Silhouette method - PAM")
# Gap statistic (10)
set.seed(111)
fviz_nbclust(df, cluster::pam, method="gap_stat") +
  labs(subtitle = "Gap statistic method - PAM")

############ CLARA ############
# Elbow method
fviz_nbclust(df, cluster::clara, method="wss") +
  labs(subtitle="Elbow method - CLARA")
# Silhouette method (3)
fviz_nbclust(df, cluster::clara, method="silhouette") +
  labs(subtitle="Silhouette method - CLARA")
# Gap statistic (2)
set.seed(111)
fviz_nbclust(df, cluster::clara, method="gap_stat") +
  labs(subtitle = "Gap statistic method - CLARA")

############ hierarchical clustering ############
# Elbow method
fviz_nbclust(df, hcut, method="wss") +
  labs(subtitle="Elbow method - hclust")
# Silhouette method (8)
fviz_nbclust(df, hcut, method="silhouette") +
  labs(subtitle="Silhouette method - hclust")
# Gap statistic (10)
set.seed(111)
fviz_nbclust(df, hcut, method="gap_stat") +
  labs(subtitle = "Gap statistic method - hclust")
