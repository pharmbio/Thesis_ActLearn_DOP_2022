import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib import pyplot
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


X, y = make_blobs(n_samples=200, centers=3, cluster_std=[10.0, 2, 3], random_state=22, n_features=2)
X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2, random_state=1)


print(X.shape)
#pyplot.scatter(X[:,1], y)
pyplot.scatter(X[:,0], X[:,1], c=y)
#plt.legend(["class 0", "class 1", "class 2"], loc='upper left')
pyplot.show()

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, .1, 1e-2],
                         'C': [1, 10, 100]}
                        ]
clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5)
clf.fit(X_train, y_train)



PredProb = clf.predict_proba(X_test)
predicted_class = clf.predict(X_test)
print(predicted_class)
#print(PredProb)

pyplot.scatter(X_test[:,0], X_test[:,1], c=predicted_class)
pyplot.show()
