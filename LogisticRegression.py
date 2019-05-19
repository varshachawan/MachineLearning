# Chawan,Varsha Rani
# 1001553524
# Logistic Regression
from math import exp

def predict(row, coefficients):
    #calculates the sigmoid value , actual value of output
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))

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
    for i in range(len(testdata) - 1):
        yhat += coef[i + 1] * testdata[i]
    sig = 1.0 / (1.0 + exp(-yhat))
    yhR = round(sig)
    return (yhR)

def normalize_dataset(trainingData):
    # Find minimum and maximum of dataset to normalise the data
    minmax = list()
    for i in range(len(trainingData[0])):
        col_values = [row[i] for row in trainingData]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    for row in trainingData:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def main():
    trainingSet = [[170, 57, 32, 0], [192, 95, 28, 1], [150, 45, 30, 0], [170, 65, 29, 1],
                   [175, 78, 35, 0], [185, 90, 32, 1], [170, 65, 28, 0], [155, 48, 31, 1],
                   [160, 55, 30, 0], [182, 80, 30, 1], [175, 69, 28, 0], [180, 80, 27, 1],
                   [160, 50, 31, 0], [175, 72, 30, 1]]

    testSet = [[155, 40, 35],[170, 70, 32],[175, 70, 35],[180,90,20]]

    normalize_dataset(trainingSet)
    normalize_dataset(testSet)

    l_rate = 0.1
    n_epoch = 10
    coef = coefficients_sgd(trainingSet,l_rate,n_epoch)
    print("******************** (2a)************************")
    print("The thetha(coefficients) values are :", coef)

    # logic to predict the class for Test data
    print("******************** (2b)***********************")
    for row in testSet :
        res = Test(row,coef)
        if res == 0 :
            print("The Predicted class for the dataSet :", row , "is", "'W'")
        else:
            print("The Predicted class for the dataSet :", row, "is", "'M'")

main()