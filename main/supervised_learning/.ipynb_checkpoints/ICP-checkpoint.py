from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np


def computeConformityScores(predProb, y_values):
    nrCases, nrLabels = predProb.shape
    category = np.unique(y_values.astype(int))
    calibLabels = y_values.astype(int)
    MCListConfScores = []  # Class wise List of conformity scores
    for i in range(0, nrLabels):
        clsIndex = np.where(calibLabels == category[i])
        classMembers = predProb[clsIndex, i]
        MCListConfScores.append(classMembers[0]) #MCListConfScores[i]+ classMembers.tolist()[0]

    return MCListConfScores


def computePValues(MCListConfScores, testConfScores):
    nrTestCases, nrLabels = testConfScores.shape
    pValues = np.zeros((nrTestCases,  nrLabels))

    for k in range(0, nrTestCases):
        for l in range(0, nrLabels):
            alpha = testConfScores[k, l]
            classConfScores = np.ndarray.flatten(np.array(MCListConfScores[l]))
            pVal = len(classConfScores[np.where(classConfScores < alpha)]) + (np.random.uniform(0, 1, 1) * \
                len(classConfScores[np.where(classConfScores == alpha)]))
            tempLen = len(classConfScores)
            pValues[k, l] = pVal/(tempLen + 1)

    return pValues


def computePredictionSet(MCListConfScores, testConfScores):
    nrTestCases, nrLabels = testConfScores.shape
    pValues = np.zeros((nrTestCases,  nrLabels))
    predictionSet = np.empty(nrTestCases, dtype=object)
    for i in range(nrTestCases):
        predictionSet[i] = []
    significance = 0.05

    for k in range(0, nrTestCases):
        for l in range(0, nrLabels):
            alpha = testConfScores[k, l]
            classConfScores = np.ndarray.flatten(np.array(MCListConfScores[l]))
            pVal = len(classConfScores[np.where(classConfScores < alpha)]) + (np.random.uniform(0, 1, 1) * \
                len(classConfScores[np.where(classConfScores == alpha)]))
            tempLen = len(classConfScores)
            pValues[k, l] = pVal/(tempLen + 1)
            if pValues[k, l] > significance:
                predictionSet[k].append(l)
    return predictionSet


#X, y = make_blobs(n_samples=100, centers=3, n_features=2)
n_samples = 200
X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=[10.0, 2, 3], random_state=22, n_features=2)

X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_calib, y_train, y_calib \
    = train_test_split(X_train, y_train, test_size=.3, random_state=1)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, .1, 1e-2],
                         'C': [1, 10, 100]}
                        ]
clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5)
clf.fit(X_train, y_train)


calibPredProb = clf.predict_proba(X_calib)
testPredProb = clf.predict_proba(X_test)

srcMCListConfScores = computeConformityScores(calibPredProb, y_calib)
#pValues = computePValues(srcMCListConfScores, testPredProb)
predSet = computePredictionSet(srcMCListConfScores, testPredProb)

print(predSet)