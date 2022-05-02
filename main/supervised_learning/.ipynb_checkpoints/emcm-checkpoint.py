"""EXPECTED MODEL CHANGE MAXIMISATION FOR ACTIVE LEARNING IN LR
Input:
X_labeled = small labeled data set (called D in the paper) with n points
y_labeled
X_pool = the unlabelled pool set
linear_model = the linear regression model (called f(x;theta)) in the paper
K_members = number of regressors in the ensemble

Ouput:
x_star_idx = the instance's index to be sampled for active learning
x_star_val = the vector of features of the sampled instance
"""
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import cross_val_score, KFold, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report, f1_score, mean_squared_log_error, recall_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn import svm, tree    #https://scikit-learn.org/stable/modules/svm.html
                                 #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def emcm_query(X_labeled, y_labeled, X_pool, linear_model, K_members):
    
    # 0. Train the linear regresor in X_labeled to build f(x;theta)
    fx = linear_model.fit(X_labeled, y_labeled)
    
    # 1. Construct an ensemble with boostrap examples
    Bk = AdaBoostRegressor(base_estimator = linear_model, n_estimators = K_members, random_state=5563).fit(X_labeled, y_labeled)
    
    # 2. for each x_candidate in x_pool do
    modelChange_per_candidate = np.zeros(X_pool.shape[0])
    for x_candidate_idx, x_candidate_value in enumerate(X_pool):
        gradient_per_candidate = []
        # 3. for each member in the ensemble
        for k in range(K_members):
            
            # 4. y_k(x_candidate_value) = f_k(x_candidate_value)
            yk_candidate = Bk.estimators_[k].predict([x_candidate_value]) 
            
            # 5.Calculate the derivative using Eq.13:
            delta_lk = (fx.predict([x_candidate_value]) - yk_candidate) * x_candidate_value
            gradient_per_candidate.append(delta_lk)
        
        # 7. Estimate the true model change by expectation calculation over K possible labels with Eq.14
        modelChange_per_candidate[x_candidate_idx] = (1/K_members)*np.sum(np.linalg.norm(gradient_per_candidate))
        
    # 8.Select the x that maximises the change
    x_star_idx = np.argmax(modelChange_per_candidate)
    x_star_value = X_pool[x_star_idx]
    
    return x_star_idx, x_star_value