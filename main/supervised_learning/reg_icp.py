import numpy as np
import sys
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from math import exp, ceil, pi
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Compute normalized conformity scores
def computeConformityScores(pred, y):
    res = np.abs(y - pred)
    return res

# Compute confidence intervals
def computeInterval(confScore, testPred, epsilon = 0.1):
    if confScore is None:
        sys.exit("\n NULL model \n")
    confScore = np.sort(confScore)
    nrTestCases  = len(testPred)
    intervals = np.zeros((nrTestCases,  2))

    for k in range(0, nrTestCases):
        # Compute threshold for split conformal, at level alpha.
        n = len(confScore)

        if (ceil((n) * epsilon) <= 1):
            q = np.inf
        else:
            q= (confScore[ceil((n) * (1 - epsilon))])

        intervals[k, 0] = testPred[k] - q
        intervals[k, 1] = testPred[k] + q

    return intervals

def fit_any_model(X_train, y_train, X_calib, y_calib, testData, model):
    model.fit(X_train, y_train)
    train_predict = model.predict(X_train)
    calibPred = model.predict(X_calib)
    testPred = model.predict(testData)

    return train_predict, calibPred, testPred

def ICPRegression(X_train, y_train, X_calib, y_calib, X_test, model, returnPredictions = False):
    if (X_train is None) or (X_calib is None):
        sys.exit("\n 'training set' and 'calibration set' are required as input\n")

    epsilon = 0.1
    train_predict, calib_pred, testPred = fit_any_model(X_train, y_train, X_calib, y_calib, X_test, model)

    confScores = computeConformityScores(calib_pred, y_calib)
    intervals = computeInterval(np.sort(confScores), testPred, epsilon)

    if returnPredictions:
        return calib_pred, testPred

    return intervals, testPred