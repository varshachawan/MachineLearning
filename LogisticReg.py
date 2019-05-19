# Chawan,Varsha Rani
# 1001553524
# Logistic Regression
from math import exp

def Accuracy(y,yhat):
    correct = 0
    for i in range(len(y)):
        if y[i] == yhat[i]:
            correct += 1
    acc =(correct / float(len(y))) * 100
    return acc


def testModel(testSet,thetha):
    res_base = []
    Base_model_pred = []
    actual = []
    for row in testSet:
        actual.append(row[-1])
        res , ignore = Test(row[:3], thetha)
        res_base.append(res)
        if res == 0:
            pred = 'W'
        else:
            pred = 'M'
        Base_model_pred.append(pred)
    return actual , Base_model_pred,res_base

def predict(row, coefficients):
    #calculates the sigmoid value , actual value of output
    yhat = coefficients[0]
    for i in range(len(row)-1):
         yhat += coefficients[i + 1] * row[i]
    return (1.0 / (1.0 + exp(-yhat)))

def coefficients_sgd(train, l_rate, n_epoch):
    # calculates coefficients using SGD
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    return coef


def Test( testdata,coef) :
# predicts the actual value of  Y for each test data
    yhat = coef[0]
    for i in range(len(testdata)):
        yhat += coef[i + 1] * testdata[i]
    sig = (1.0 / (1.0 + exp(-yhat)))
    yhR = round(sig)
    return yhR , sig

def normalize_dataset(trainingData):
    # Find minimum and maximum of dataset to normalise  the data
    minmax = list()
    for i in range(len(trainingData[0])):
        col_values = [row[i] for row in trainingData]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    for row in trainingData:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
