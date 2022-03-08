%load_ext autoreload
%autoreload 2

from pandas import read_excel
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random
import timeit
import pickle

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm, tree    #https://scikit-learn.org/stable/modules/svm.html
                                 #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

from modAL.models import ActiveLearner, Committee             #https://modal-python.readthedocs.io/en/latest/content/models/ActiveLearner.html
from modAL.uncertainty import entropy_sampling     #https://modal-python.readthedocs.io/en/latest/content/apireference/uncertainty.html

import functions as fun

# Loading data sets
full_data_BatchA = pd.read_csv('/home/jovyan/Thesis_ActLearn_DOP_2022/main/active_learning/data/full_data_BatchA.csv')
y = full_data_BatchA['Label'].to_numpy()
X_morgan = full_data_BatchA.drop(['Label'], axis = 1).to_numpy()

# Creating classifiers with only default values exept random state for replications
svm_clf1 = svm.SVC(random_state = 0, probability=True)
knn_clf1 = KNeighborsClassifier(n_jobs=-1)
rf_clf1 = RandomForestClassifier(max_depth=10, random_state=0, n_jobs = -1)
ada_clf1 = AdaBoostClassifier(n_estimators=10, random_state=0)

# Parameters for ML model
train_size = 0.1
test_size = 0.3

# ------------------- ACTIVE LEARNING -------------------
 
# Parameters for AL
N_QUERIES = int(2*len(x_pool)/3)
N_MEMBERS = len(Classifiers)

#Creating the committee
# a list of ActiveLearners:
learners = [svm_clf1, knn_clf1, rf_clf1]

committee = Committee(
    learner_list=learners,
    query_strategy=vote_entropy_sampling
)

#Training with query by committee
cf_mat_x_clssifr = []
prfmc_x_classifr = []
for idx in range(n_queries):
    query_idx, query_instance = committee.query(X_pool)
    # Teach our ActiveLearner model the record it has requested.
    XX, yy = x_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
    committee..teach(X=XX, y=yy)
    
    # Remove the queried instance from the unlabeled pool.
    x_pool, y_pool = np.delete(x_pool, query_index, axis=0), np.delete(y_pool, query_index)
    
    y_pred = committee.predict(x_test)
    model_accuracy = committee.score(x_test, y_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    #print(cf_matrix)
    performance_history.append(model_accuracy)
    cf_matrix_history.append(cf_matrix)
    
    if index % 100 == 0:
            print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        

    
