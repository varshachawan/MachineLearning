# Chawan,Varsha Rani
# 1001553524
# Boosting


import numpy as np
from math import exp


def reclassify(y_pred):
    #Reclassifying zeros as minus ones
    Y_pred_new = []
    for data in y_pred:
        if data == 1:
            Y_pred_new.append(data)
        else:
            Y_pred_new.append(-1)
    return Y_pred_new


def ErrorRate(y,yhat):
    # calculate the Error rate for each set of ensembles
    correct = 0
    for i in range(len(y)):
        if y[i] == yhat[i]:
            correct += 1
    err = (1 - (correct / float(len(y))))*100
    return err


def finalPredictions(alpha_L,y_pred_train_L):
    # calculate the weighted predictions
    final_pred = np.zeros(len(y_pred_train_L[0]))
    for alpha,y_pred in zip(alpha_L,y_pred_train_L):
        y_pred = reclassify(y_pred)
        prod_train =[]
        for y in y_pred:
            prod_train.append( y*alpha)
        list_sum = []
        for x,y in zip(final_pred,prod_train):
            list_sum.append(x+y)
        final_pred = list_sum
    final_pred = np.sign(final_pred)
    return final_pred


def predictlogistic(X_Train,weights):
    # Predicts the labels using logistic regression classifier
     y_pred = []
     for X_row in X_Train:
         yhat = weights[0]
         for i in range(len(X_row)):
             yhat += weights[i + 1] * X_row[i]
         sig = 1.0 / (1.0 + exp(-yhat))
         yhR = round(sig)
         y_pred.append(yhR)
     return y_pred


def predict(x_row, coefficients):
    #calculates the sigmoid value , actual value of output
    yhat = coefficients[0]
    for i in range(len(x_row)):
        yhat += coefficients[i + 1] * x_row[i]
    return 1.0 / (1.0 + exp(-yhat))


def logisticRegression(X_Train,Y_Train,weights):
    # calculates coefficients using SGD, Logistic Regression classifier
    coef = [0.0 for i in range(len(X_Train[0])+1)]
    l_rate = 0.1
    n_epoch = 10
    for epoch in range(n_epoch):
        for x_row ,y_row , weight in zip(X_Train,Y_Train,weights):
            yhat = predict(x_row ,coef)
            error = (y_row - yhat)*weight
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(x_row)):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * x_row[i]
    return coef


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


def readData(filename):
    # Reads text file and converts to array of list
    trainData = []
    with open(filename, 'r') as file:
        for line in file:
            data = []
            splited = line.split(",")
            len_line = len(splited)
            for i in range(len_line):
                elem = float(splited[i])
                data.append(elem)
            trainData.append(data)
    return trainData


def boosting(X_Train,Y_Train,X_Test,n_classifiers):
    # Applying boosting for N ensembles
    n_train = len(X_Train)
    hypothesis_L = []
    alpha_L = []
    y_pred_train_L = []
    y_pred_test_L = []

    # Initialise weights
    Weights = np.ones(n_train) / n_train

    for i in range(n_classifiers):
        hypothesis = logisticRegression(X_Train,Y_Train,Weights)
        y_pred_train = predictlogistic(X_Train,hypothesis)
        y_pred_test = predictlogistic(X_Test,hypothesis)
        hypothesis_L.append(hypothesis)
        y_pred_test_L.append(y_pred_test)
        y_pred_train_L.append(y_pred_train)
        # wrongly predicted records
        wrong = []
        for y, yh in zip(Y_Train,y_pred_train):
            x = (y!=yh)
            wrong.append(int(x))
        yyh = []
        for w in wrong :
            if w ==1:
                yyh.append(-1)
            else :
                yyh.append(w)
        # calculate epsilon
        err_m = np.dot(Weights, wrong) / sum(Weights)
        #calculate alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        # Update Weights
        Weights = np.multiply(Weights, np.exp([-float(x) * alpha_m for x in yyh]))
        Weights = (1.0/sum(Weights))*Weights

        alpha_L.append(alpha_m)
    print("hypothesis List :", hypothesis_L)
    return alpha_L,y_pred_train_L,y_pred_test_L


def main():
    # Main logic to implement Boosting
    trainData = readData("train.txt")
    testData = readData("test.txt")
    normalize_dataset(trainData)
    normalize_dataset(testData)
    X_Train = []
    Y_Train = []
    X_Test = []
    Y_Test = []
    for data in trainData:
        X_Train.append(data[:-1])
        Y_Train.append(data[-1])

    for data in testData:
        X_Test.append(data[:-1])
        Y_Test.append(data[-1])

    Y_Train_new = []
    for data in Y_Train :
        if data == 1 :
            Y_Train_new.append(data)
        else:
            Y_Train_new.append(-1)

    Y_Test_new = []
    for data in Y_Test:
        if data == 1:
            Y_Test_new.append(data)
        else:
            Y_Test_new.append(-1)

    for n_classifiers in [1, 10, 25, 50]:

        alpha_L, y_pred_train_L, y_pred_test_L = boosting(X_Train,Y_Train,X_Test,n_classifiers)

        final_train_pred = finalPredictions(alpha_L,y_pred_train_L)
        final_test_pred = finalPredictions(alpha_L,y_pred_test_L)

        err_train = ErrorRate(Y_Train_new, final_train_pred)
        err_Test = ErrorRate(Y_Test_new,final_test_pred)
        # print("==========================================================================")
        print("For", n_classifiers, "classifiers, Error rate for train data is ", err_train)
        print("For", n_classifiers, "classifiers, Error rate for test data is ", err_Test)
        print("==========================================================================")

main()