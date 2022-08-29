# Classification-and-Clustering
This project consists of two parts: the first part is a classification problem, and the second is a clustering problem.

For the classification task, various supervised machine learning techniques are used to perform the classification of the data set and include Logistic Regression (LR), Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), Random Forest, Boosting, Bagging and Naive Bayes. Since the data is high-dimensional (500 features with only 400 observations), we can either directly use methods that deal with high-dimensional data, or we can perform feature selection first, then apply classification algorithms to the lower-dimension data. To estimate the test error, we use cross-validation. If there are tuning parameters that need to be estimated, nested cross-validation is applied to ensure that the estimation of parameters and test errors are independent of each other.

The clustering algorithms considered include K-means, PAM, CLARA, and Hierarchical Clustering.

#Classification Accuracy - 84%

#Optimal Clusters - 8
