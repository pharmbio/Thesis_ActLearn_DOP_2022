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
from modAL.disagreement import vote_entropy_sampling     #https://modal-python.readthedocs.io/en/latest/content/apireference/uncertainty.html

import functions as fun

# Loading data sets
full_data_BatchA = pd.read_csv('C:/Users/dalia/PycharmProjects/Thesis_DS_2022/Thesis_UU/My_examples/full_data_BatchA.csv')
y = full_data_BatchA['Label'].to_numpy()
X_morgan = full_data_BatchA.drop(['Label'], axis = 1).to_numpy()

# Creating classifiers with only default values exept random state for replications
svm_clf1 = svm.SVC(random_state = 0, probability=True)
knn_clf1 = KNeighborsClassifier(n_jobs=-1)
rf_clf1 = RandomForestClassifier(max_depth=10, random_state=0, n_jobs = -1)
ada_clf1 = AdaBoostClassifier(n_estimators=10, random_state=0)

# Parameters for ML model
train_size = 0.05
test_size = 0.3

#Generating test, training and pool
x_train, y_train, x_test, y_test, x_pool, y_pool = fun.split(x_dataset=X_morgan, y_dataset= y,
                                                             ini_train_size= train_size, test_size=test_size)

# ------------------- ACTIVE LEARNING -------------------


def initialise_committe(x_train, y_train,classifiers_list, query_str):
    # initializing Committee members
    n_members = len(classifiers_list)
    learner_list = list()

    for member_idx in range(n_members):

        # initializing learner
        learner = ActiveLearner(
            estimator=classifiers_list[member_idx],
            X_training=x_train, y_training = y_train,
            query_strategy = query_str
        )
        learner_list.append(learner)

    # assembling the committee
    committee = Committee(learner_list=learner_list)
    return committee


# Active learning with QBC
def active_learnig_train_committee(n_queries, x_train, y_train, x_test, y_test, x_pool, y_pool, classifiers_list, query_str):
    performance_history = []
    cf_matrix_history = []

    committee = initialise_committe(x_train,y_train,classifiers_list,query_str)
    # Making predictions
    y_pred = committee.predict(x_test)

    # Calculate and report our model's accuracy.
    model_accuracy = committee.score(x_test, y_test)

    # Generate the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    # Save our model's performance for plotting.
    performance_history.append(model_accuracy)
    cf_matrix_history.append(cf_matrix)

    for index in range(n_queries):
        # Query a new instance
        query_idx, query_instance = committee.query(x_pool)

        # Teach our ActiveLearner model the record it has requested.
        XX, yy = x_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1, )
        committee.teach(X=XX, y=yy)

        # Remove the queried instance from the unlabeled pool.
        x_pool, y_pool = np.delete(x_pool, query_idx, axis=0), np.delete(y_pool, query_idx)

        y_pred = committee.predict(x_test)
        model_accuracy = committee.score(x_test, y_test)
        cf_matrix = confusion_matrix(y_test, y_pred)
        performance_history.append(model_accuracy)
        cf_matrix_history.append(cf_matrix)

        if index % 100 == 0:
                print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        
    return performance_history , cf_matrix_history, committee


# Parameters for AL
N_QUERIES = 50 # int(2*len(x_pool)/3)
# Creating the committee form a list of ActiveLearners:
learners1 = [svm_clf1, knn_clf1, rf_clf1]
committiees = [learners1]
query_strategy = vote_entropy_sampling

# Timer
tic = timeit.default_timer()

cf_mat_x_classifr = []
save = False
count = 0
for lista in committiees:

    print(f'Training with committee {list}')

    _, cf_mat_his, committee = active_learnig_train_committee(n_queries = N_QUERIES, x_train=x_train, y_train = y_train,
                                                              x_test = x_test, y_test = y_test, x_pool = x_pool, y_pool = y_pool,
                                                              classifiers_list = lista, query_str = query_strategy)
    cf_mat_x_classifr.append(cf_mat_his)

    # Saving model
    if save:
        filename = "".join(["home/jovyan/Thesis_ActLearn_DOP_2022_/main/active_learning/qbc_entropy_default_models/committe",str(count),".sav"])

    count += 1

print('Finally done!!!')

    
