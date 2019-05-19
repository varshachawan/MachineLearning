# Chawan,Varsha Rani
# 1001553524
# Bagging

from random import randrange
import logisticReg as model
from math import exp


def predict(row, coefficients):
    #calculates the sigmoid value , actual value of output
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))


def bootstrapSample(traindata_orignal):
    # Create random train data using bootstrap technique
    bootstrap_data = []
    len_data = len(traindata_orignal)
    for i in range(len_data-1) :
        index = randrange(len_data)
        bootstrap_data.append(traindata_orignal[index])
    return bootstrap_data


def ErrorRate(y,yhat):
    #calculate Accuracy and ErrorRate
    correct = 0
    for i in range(len(y)):
        if y[i] == yhat[i]:
            correct += 1
    err = (1 - (correct / float(len(y))))*100
    return err


def Test( testdata,coef_L) :
# predicts the actual value of  Y for each test data by taking
#  maximum of N bags
    yh_bag = []
    for coef in coef_L :
        yhat = coef[0]
        for i in range(len(testdata) - 1):
            yhat += coef[i + 1] * testdata[i]
        sig = 1.0 / (1.0 + exp(-yhat))
        yhR = round(sig)
        yh_bag.append(yhR)
    yhat_max = max(yh_bag, key=yh_bag.count)
    return yhat_max


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

def main():
    #Bagging for 1,10,50,100 classifiers for logistic regression
    trainData_orignal = readData("train.txt")
    testData = readData("test.txt")
    model.normalize_dataset(testData)
    for n_classifiers in [1,10,50,100]:
        coeff_list = []
        for i in range(n_classifiers):
            if n_classifiers == 1 :
                B_data = trainData_orignal
            else :
                B_data = bootstrapSample(trainData_orignal)
            coeff = model.logistic(B_data)
            coeff_list.append(coeff)
        print("coeff_list : ",coeff_list)
        y_list=[]
        yhat_List =[]
        # predicting Test data
        for row in testData:
            y = row[-1]
            yhat = Test(row, coeff_list)
            yhat_List.append(yhat)
            y_list.append(y)
        err = ErrorRate(y_list,yhat_List)
        if n_classifiers ==1:
            n_classifiers = "Base Model"
        # print("==========================================================================")
        print("For",n_classifiers,"classifiers, Error rate is ",err)
        print("==========================================================================")

main()