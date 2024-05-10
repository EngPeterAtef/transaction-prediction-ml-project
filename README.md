
# Santander Customer Transaction Prediction

## Problem Definition
The problem is a binary classification task aimed at predicting whether a customer will make a specific transaction in the future. Given historical data on customer transactions, Santander seeks to develop a predictive model to identify customers likely to engage in the specified transaction, regardless of the transaction amount.

## Motivation
Santander's motivation for addressing this problem stems from its mission to help people and businesses prosper by providing tailored financial solutions. By accurately predicting customer transactions, Santander can optimize its marketing strategies, offer targeted product recommendations, and enhance customer engagement. Additionally, by leveraging machine learning algorithms, Santander aims to improve the efficiency and effectiveness of its operations, ultimately leading to better customer outcomes and business performance.

## **Models**
1. Logistic Regression
2. Random Forest
3. Adaboost Classifier
4. SVM
5. XGB Classifier

## Evaluation Metrics:
<ul>
<li>
  Weighted F1 score
</li>
<li>Accuracy</li>
<li>Micro and Macro precision and recall</li>
</ul>

## Exploratory Data Analysis (EDA)
### Links to the dataset: 
https://www.kaggle.com/c/santander-customer-transaction-prediction/data
**The dataset size is 100K and the number of features is 200**
### Target Variable Analysis
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/d330be2c-51c3-4088-ac83-28bc5144f47a)
We can see that the number of points in class zero is way larger than the number of points in class one.
### Features Distribution
The following graphs show the distribution of the features:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/4e0611ca-f98a-406c-8858-7a221d6b9bb7)
We can see that all features are normally distributed so using this information we can standardize the features by applying the following equation for each feature independently: X = (X - mean) / std
so that the mean of each feature will be almost zero and the variance almost one.
###  Features Correlation
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/2ae1f503-fcd3-4703-811c-01e9a9e4d8d0)
We can see that all the features are almost independent of each other.
### Variance Thresholding
After getting the variance of each feature the mean of variances will be 0.999, using this value we will threshold the features by their variance and the result is getting only 87 features out of 200. The following graph shows the correlation between the 87 features:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/0d53d8f8-43f7-4e98-b588-ea7dfba84657)
We can notice that the features are still independent of each other.
### PCA
We applied PCA to get n components where 20<=n<=180 and we got the follow	ing results:
For n = 20
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/9cee8e8b-64eb-4ad5-b023-a8ed9bf69be7)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/0c4ae86d-9df1-46e3-b577-f923693a5f6b)
For n = 100
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/62d5334d-de21-41d8-a296-73d8b4ee14d3)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/97e030d4-f941-4ac8-9d53-78ed2f7c274c)
From the previous analysis, we can conclude that most of the variance ratio is concentrated in the first 20 features to maximize the number of points we can use for training we will use the minimum number of features because there will not be a huge loss in variance ratio.
**Note** that we needed to reduce the dimensionality of the space for 2 reasons:
<ol>
  <li>
    To avoid the problem of the curse of dimensionality
  </li>
  <li>
    It takes a lot of time for SVM and ensemble models to train and apply cross-validation algorithms and we want to maximize the number of points we use so we need to reduce the number of features to achieve that.
  </li>
</ol>
<b>So our final dataset used for training is 100K rows and 20 features, each feature is normally distributed.</b>

## Models Analysis
All models are validated using cross-validation with 10 folds.
### Base Model
<ol>
  <li>Stratified</li>
  
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/d5fe16a0-e1bb-497c-ba67-33c0a6961cbd)
<li>most_frequent</li>

![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/dde8a38b-f1ab-46d4-87fc-40dafcbf983d)
<li>uniform</li>

![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/5caab6c0-34c1-4270-92c7-a4c76aa02aeb)
<li>constant</li>

![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/52b85996-1e83-411c-ad6a-948cad946da0)

<li>prior</li>

![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/6cb1e902-167d-4821-a12c-30e97197c824)

</ol>

### Logistic Regression
Logistic regression is a powerful tool for predicting categorical outcomes. It is used in a wide variety of fields, including marketing, medicine, and finance. For example, logistic regression can be used to predict the likelihood that a customer will buy a product, the likelihood that a patient will develop a disease or the likelihood that a company will go bankrupt.

#### Feature Importance Plot
Note that we now only use 20 features that we got from PCA to be able to train and apply cross-validation on all models.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/a8a63d42-8dce-40e4-8786-4aade902ad2d)
Based on the previous plot we can see that the first feature is the most important one with 75% importance and the other 19 features have an importance of 25%. So we may reduce the number of features to ten instead of twenty if you will use logistic regression to solve this problem.
#### Partial Dependence Analysis
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/2ee1c2a8-cc21-4fc2-b7fa-0d0c6b958a93)
We can see that for the first feature the relationship isn’t linear but for the rest of the features, it’s a linear relationship either direct or inversely proportional.
#### Learning Curves Plot
shows the training error (Ein) and validation error (Eval) as a function of the training set size
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/73f66027-30e4-4abb-b7c9-48ad4b75a698)
We can see that the training error is near to the validation error so the model can generalize pretty well.
#### Hyperparameter Tuning
This is a process of adjusting the parameters of a model to optimize its performance. It can be done using techniques like grid search, random search, or Bayesian optimization.
**Grid Search**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/cb1aee09-5f3f-4aca-8e88-6865d34e53e5)
Here are some heatmap visualizations of the grid search results:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/1f943f2e-a133-4e96-99f1-3b145af8f537)
After applying the grid search we found that the best parameters are: 
{'C': 0.5, 'penalty': 'l2', 'solver': 'lbfgs'}
After training a model using the best parameters we got from the search grid we got the following results:
**Confusion matrix:**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/2c6fa655-1a24-439d-abd9-68f36d4ef1e4)
**Classification report:**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/610c954c-cebd-4aa2-8857-cf4fcca8d605)
**Train-Validation Curve**
Here are some Train-Validation Curves that we further used for the hyperparameter tuning process:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/33927dfe-1722-4b80-ad49-980242fdddde)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/c8882680-6c81-4283-8286-af919704fa0a)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/405c159f-e6f5-40c6-b2c3-bbd9aa8e752b)
#### Bias-variance Analysis
●	mean square error:  0.0973067
●	bias:  0.096430338
●	var:  0.0008763620000000017
●	Estimated Eout:  0.09730670000000001
### Support Vector Machines
Support Vector Machines (SVMs) are a class of supervised learning algorithms used for classification and regression analysis. SVMs work by finding the hyperplane that best separates the data into different classes. The optimal decision boundary is the hyperplane that maximizes the margin between the two classes. In the case where the data is not linearly separable, SVMs use a kernel trick to map the data into a higher-dimensional space where the data can be linearly separated.
#### Feature Importance Plot
Note that we now only using 20 features that we got from PCA to be able to train and apply cross-validation on all models.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/94284429-e50c-4ede-89d3-0e4db9288e07)
Based on the previous plot we can see that the 18th feature is the highest importance and the more we add features, the less importance they get. Unlike logistic regression, SVM makes use of most of the features.
#### Partial Dependence Analysis
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/13b26887-2332-4d2e-9d64-6f875c80b525)
We can see that for the first feature the relationship isn’t linear but for the rest of the features, it’s a linear relationship either direct or inversely proportional.
#### Learning Curves Plot
shows the training error (Ein) and validation error (Eval) as a function of the training set size
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/aa01e13a-5a84-4a0f-ac65-228f73fe48c9)
#### Hyperparameter Tuning
This is a process of adjusting the parameters of a model to optimize its performance. It can be done using techniques like grid search, random search, or Bayesian optimization.
**Grid Search**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/312f19c4-7e80-4236-be29-2fd12b3c7bf3)
Here are some heatmap visualizations of the grid search results:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/cfdec9cb-d257-43a6-9ce6-02aa7c2b3e0b)
After applying the grid search we found that the best parameters are: 
{'max_iter': 1000, 'random_state': 0, 'C': 1, 'gamma': 1, 'kernel': 'rbf'}
After training a model using the best parameters we got from the search grid we got the following results:
**Confusion matrix:**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/006a6cd6-f638-46fc-be0e-09eaec3ed1f7)
**Classification report:**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/f422efd7-050b-4674-9e18-063012084442)
**Train-Validation Curve**
Here are some Train-Validation Curves that we further used for the hyperparameter tuning process:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/94732cc5-a0de-4bf7-9980-7aaaeb85275f)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/9348921f-c28c-4859-ac6c-ce110e072da2)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/a980225c-fb25-4667-83d6-64480144fbc0)
#### Bias-variance Analysis
●	mean square error:  0.1202861
●	bias:  0.10426006449999999
●	var:  0.016026035499999983
●	Estimated Eout:  0.12028609999999997
### Random Forest
Random Forest is a popular machine learning algorithm that falls under the category of ensemble learning methods. It is a type of decision tree algorithm that generates multiple decision trees and combines their predictions to produce the final output.
#### Feature Importance Plot
Note that we now only use 20 features that we got from PCA to be able to train and apply cross-validation on all models.
A feature importance plot shows the importance of each feature in the model. It can be used to identify the most important features and to understand the impact of each feature on the model's predictions.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/9e4d0c05-157b-4bf4-addc-2835a7409b6a)
We can see that the first feature is the most important one and the other features are almost equally important.
#### Learning Curves Plot
shows the training error (Ein) and validation error (Eval) as a function of the training set size.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/2021043e-029d-4f29-a044-72111509f02e)
Even if the error on the training is zero, there is proof that shows that random forest isn’t overfitting even Ein = 0 and you can see that the error on validation isn’t big even if it’s far from Ein.
#### Partial Dependence Plot
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/63e898fd-3f76-42b7-baae-6b5d3cce8981)
We can see that the relationship between the model and each feature is non-linear except for the first feature has a linear, directly proportional relationship with the model.
#### Hyperparameter Tuning
This is a process of adjusting the parameters of a model to optimize its performance. It can be done using techniques like grid search, random search, or Bayesian optimization.
**Number of estimators effect**
The following graph shows the effect of changing the number of estimators when the other parameters are constants.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/f805ef2b-a177-4b88-9b8c-3bd6a8ece784)
We can see that for our problem using a lot of estimators isn’t a good thing because that will lead to overfitting (bad generalization).
**Grid Search**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/7ca1a8ec-861f-4891-957c-ce7ba261789d)
Here are some heatmap visualizations of the grid search results:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/6ca3db00-5411-4fe0-89fe-dcfa31c5bae5)
After applying the grid search we found that the best parameters are: 
{'max_depth': 50, 'min_samples_split': 5, 'n_estimators': 20}
After training a model using the best parameters we got from the search grid we got the following results:
**Confusion matrix:**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/261467b2-149b-4ce1-9ce4-bcc055532eb9)
**Classification report:**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/ea589554-a3f8-4637-9621-a69885432084)
**Train-Validation Curve**
Here are some Train-Validation Curves that we further used for the hyperparameter tuning process:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/46271c78-f9eb-4f7b-aa64-41f8655451d3)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/43023def-9306-4f80-995e-f4f6cc5d8600)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/c72571c8-5514-463e-aee1-bb01add57f2b)
#### Bias-variance Analysis
●	mean square error:  0.10085995
●	bias:  0.08931127525000002
●	var:  0.011548674750000005
●	Estimated Eout:  0.10085995000000002
#### Tree Plot
A tree plot shows the structure of the decision trees used in the random forest. It can be used to understand how the model makes predictions.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/16210683-9907-4331-a112-c3cd4e810a37)
**Variable importance**
The plot can show which variables (or features) are the most important in making the predictions. The importance of a feature is determined by how much the tree nodes that use that feature reduce impurity (i.e., increase homogeneity). In our case the most important feature is pca0
**Interactions between features**
The plot can show how different features interact with each other to make predictions. For example, if two features are highly correlated, the plot can show whether the random forest is consistently using one feature over the other or if it's using both in combination.
**Overfitting**
In our case, the model isn’t overfitted because there are not many shallow trees as if there are many shallow trees (i.e., with few splits) in the forest, indicating that the model is not capturing the underlying patterns in the data. This means that the model is capturing the underlying patterns in the data and has good generalization.
### Adaboost 
AdaBoost, short for Adaptive Boosting, is a popular ensemble learning algorithm that combines multiple weak learners to create a strong classifier. It operates by sequentially training a series of weak learners, such as decision trees with limited depth, and assigns higher weights to misclassified instances in each iteration. This iterative process focuses on the difficult instances, gradually improving the model's performance.
#### Feature Importance Plot
Note that we now only use 20 features that we got from PCA to be able to train and apply cross-validation on all models.
A feature importance plot shows the importance of each feature in the model. It can be used to identify the most important features and to understand the impact of each feature on the model's predictions.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/d803873c-2a8e-4392-a7ac-0140b046dc11)

We can see that the first feature is the most important one and the other features are almost equally important except for pca4, pca8, pca9, and pca17 which have zero importance.
#### Learning Curves Plot
shows the training error (Ein) and validation error (Eval) as a function of the training set size.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/01103880-6d49-4fc8-b6b2-303e08db8660)
We can see that the training error is near to the validation error so the model can generalize pretty well.
#### Partial Dependence Plot
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/19589094-899a-47dd-8921-5ba18f729ab0)
We can see that all features have a linear relationship either direct or inversely proportional.
#### Hyperparameter Tuning
This is a process of adjusting the parameters of a model to optimize its performance. It can be done using techniques like grid search, random search, or Bayesian optimization.
**Number of estimators effect**
The following graph shows the effect of changing the number of estimators when the other parameters are constants.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/55ff5487-f3d5-48d1-b241-0ef74a86ff75)
We can see that for our problem using a lot of estimators isn’t a good thing because that will increase the computations even, though there is no increase in the accuracy after 40 estimators.
**Grid Search**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/07dae3ca-af90-4315-b5c4-9bb8b6879b30)
Here are some heatmap visualizations of the grid search results:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/65ca9348-bbfa-4c42-8474-75017812f7fa)
After applying the grid search we found that the best parameters are: 
{'algorithm': 'SAMME.R', 'learning_rate': 1, 'n_estimators': 50}

After training a model using the best parameters we got from the search grid we got the following results:
**Confusion matrix:**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/4663c99a-7c73-4b41-b051-1c837168f13e)
**Classification report:**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/27ea50be-2e77-42f6-8621-668ba618bb56)
**Train-Validation Curve**
Here are some Train-Validation Curves that we further used for the hyperparameter tuning process:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/22d5a9b9-fb9c-4b92-b856-f6c4e47a8111)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/964f4e62-523c-4c2e-a406-19291325f78c)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/4d122d1a-466e-4a3b-808b-883960f54b31)
#### Bias-variance Analysis
●	mean square error:  0.1005537
●	bias:  0.10051915900000002
●	var:  3.454099999999998e-05
●	Estimated Eout:  0.10055370000000002
### Decision tree boosting (XGboost)
#### Feature Importance Plot
Note that we now only use 20 features that we got from PCA to be able to train and apply cross-validation on all models. A feature importance plot shows the importance of each feature in the model. It can be used to identify the most important features and to understand the impact of each feature on the model's predictions.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/b196a6b8-39dc-486f-832e-26a1ecb09741)
We can see that the first feature is the most important one and the graph shows that XGB makes full use of all features.
#### Learning Curves Plot
shows the training error (Ein) and validation error (Eval) as a function of the training set size.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/bd071472-8d62-4c45-8497-a0a5393c3782)
We can see that with the increase in training size, the difference between Ein and Eval decreases which means a better generalization.
#### Partial Dependence Plot
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/19f57b6c-99eb-438f-8e9b-25603b4f579a)
We can see that all features have a non-linear relationship either direct or inversely proportional.
#### Hyperparameter Tuning
This is a process of adjusting the parameters of a model to optimize its performance. It can be done using techniques like grid search, random search, or Bayesian optimization.

**Number of estimators effect**
The following graph shows the effect of changing the number of estimators when the other parameters are constants.
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/5e91c6be-9a37-469d-a0ec-73ea61e0dc11)
We can see that for our problem using a lot of estimators isn’t a good thing because after 70 estimators the accuracy decreases.
Also, notice that at a number of estimators = 50, it’s less than 40 estimators so we need the number of estimators carefully based on the problem.
**Grid Search**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/264b8e2e-1ed8-4fa7-888a-303aef3c7e9f)
Here are some heatmap visualizations of the grid search results:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/a57a2bf1-7cf5-45f6-b429-b1304a0bb452)
After applying the grid search we found that the best parameters are: 
{'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}

After training a model using the best parameters we got from the search grid we got the following results:
**Confusion matrix:**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/c9906749-be2c-4947-9922-f0a26d32b753)
**Classification report:**
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/660b518a-92a9-4872-9047-208982692c0f)
**Train-Validation Curve**
Here are some Train-Validation Curves that we further used for the hyperparameter tuning process:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/73e9d6c5-34ed-4b3a-9c62-9201b562cb73)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/6ecfbd76-9289-4e16-8a7c-e6844e8d8b07)
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/c3eca7eb-564f-42ca-baf6-0b672dcdb804)
#### Bias-variance Analysis
●	mean square error: 0.09885829999999998
●	bias:  0.092098
●	var:  0.0067603000000000055
●	Estimated Eout:  0.09885830000000001
## The problem of small weighted micro average
It’s noticed the weighted micro average is pretty small compared to the F1-score and accuracy that happened due to the problem of unbalanced classes in the dataset because the number of samples in class zero is way more than number of samples in class one, as you can see in the following chart:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/fafaf427-3464-4704-82ce-ad937e0cd1ed)
So, one of the solutions to this problem is oversampling which is increasing the number of instances in the minority class. Techniques like SMOTE (Synthetic Minority Over-sampling Technique) generate synthetic samples rather than simply duplicating existing ones. After applying SMOT, the size of the dataset increased to 359804 and the classes became balanced, as you can see in the following chart:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/37c45446-eccc-4f23-9237-8ae6696151c9)
but unfortunately, the problem of small micro average still exists and also another problem appeared which is getting a high bias value and variance equal to zero which means the model is so simple to solve the problem so I tried to use a higher order model (with more features => complex target features) but that didn’t solve the problem and the results are the following:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/e61f5b34-18a8-4509-80db-55f86ce8716c)
Still, the difference between the weighted micro F1-score and accuracy is big.
another solution we tried was undersampling (Reducing the number of instances in the majority class. However, this might lead to loss of information), after applying undersampling, the size of the dataset decreased to 40196 the classes became balanced, and as you can see in the following chart:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/2806d88b-8c0b-44e6-bddb-bc6f376a707e)
It gives results almost like oversampling but with lower accuracy!!!
Let’s try another way to solve this problem, set the weights for each class inversely proportional to the number of samples of each class.
class_weight = {0: 1/percentage_of_zeros, 1: 1/percentage_of_ones}
The results are:
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/998c44c7-6809-4e98-bcf2-f30d3638b5b4)
Still, there is a big difference!!!
Using ROC-AUC matric without sampling and without setting the class weights. We got almost the same results for all models: AUC=0.5
![image](https://github.com/Senior-year-second-semester-CMP2024/machine-learning/assets/75852529/c1493792-3be0-4453-9a8f-b9543a1e7f97)
even though the accuracy is almost 90% for all models the area under the ROC curve is 0.5 which is a small area that means the rate of the true positive rate equals the false positive rate and that happened because all our models classify the points of class one as class zero (FPR) with the same rate of classifying the points of class zeros as class zero (TPR).
