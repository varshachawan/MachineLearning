# Linear Discriminant Analysis
# Chawan,Varsha Rani
# 1001553524
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import math

def seperateClasses(Dataset):
    # Seperate the data points based on class
     X_Women_0 = []
     X_Men_1 = []
     for i in range(len(Dataset)):
         vector = Dataset[i]
         if vector[-1] == 'M':
             del vector[-1]
             X_Men_1.append(vector)
         else:
             del vector[-1]
             X_Women_0.append(vector)
     X_Women_0 = np.matrix(X_Women_0)
     X_Men_1 = np.matrix(X_Men_1)
     return X_Women_0 , X_Men_1

def calculateMean(X_zero ,X_one) :
    # Calculate mean for each class
    print("***************** Problem 3a ***************")
    mean_0 = np.sum(X_zero, axis=0) / X_zero.shape[0]
    mean_1 = np.sum(X_one, axis=0) / X_one.shape[0]
    print("Mean for class 0 (W) is :" , mean_0)
    print("Mean for class 1 (M) is :", mean_1)

    return mean_0 , mean_1

def calculatePrior(X_zero ,X_one):
    # calculate prior probabilities
    phi_1 = float(len(X_one)) / (len(X_one) + len(X_zero))
    phi_0 = 1-phi_1
    print("Phi for class 0 (W) is :", phi_0)
    print("Phi for class 1 (M) is :", phi_1)

    return phi_0 , phi_1

def calculatesigma(X_zero,X_one,mean_0,mean_1):
    # calculate covariance matrix for each class and common covariance
    sigma_0 = np.transpose((X_zero - mean_0))*(X_zero - mean_0)
    sigma_1 = np.transpose((X_one - mean_1)) * (X_one - mean_1)
    sigma = ( sigma_1 + sigma_0 ) / (X_one.shape[0]+ X_zero.shape[0])
    sigma_0 /= len(X_zero)
    sigma_1 /= len(X_one)
    print("sigma for class 0 (W) is :" , sigma_0)
    print("sigma for class 1 (M) is :", sigma_1)
    print("Common sigma", sigma)
    return sigma, sigma_0, sigma_1

def calculateProbFfunction(mu,sig,X):
    # Calculates the p(x/y) for each class
    m = mu.size
    k = (m/2)
    F = (1/(math.pow((2*math.pi),k)*math.sqrt(np.linalg.det(sig)))) * (math.exp(-0.5*np.matmul(np.matmul((X-mu),np.linalg.inv(sig)),np.transpose(X-mu))))
    return F

def determineClass(mu0,mu1,var0,var1,var,phi0,phi1,testData) :
    print("************Part (3b) ***********" )
    # Claculates the posterior prob and determines class
    for row in testData:
        F0 = calculateProbFfunction(mu0,var,row)
        F1 = calculateProbFfunction(mu1,var,row)
        p0 =F0*phi0
        p1 =F1*phi1
        if p0 > p1 :
            print("The predicted Gender for Data:", row ," is ", "W")
        else:
            print("The predicted Gender for Data:", row, " is ", "M")

def generateFeatures(u0,u1,s0,s1,phi0,phi1):
    print("*************** (3c) ***********")
    # generates the sample data
    u0_1D = np.ravel(u0)
    sample_W = (np.random.multivariate_normal(u0_1D,s0,50)).astype(int)
    print("Generated samples for class W :", sample_W)
    u1_1D = np.ravel(u1)
    sample_M = (np.random.multivariate_normal(u1_1D,s1,50)).astype(int)
    print("Generated samples for class M :", sample_M)
    return sample_W,sample_M

    # Mat1 = np.matmul((u1-u0), np.linalg.inv(s0))
    # print("Mat1",Mat1)
    # Mat2 = (np.matmul(np.matmul(u1,np.linalg.inv(s0)),np.transpose(u1)))-(np.matmul(np.matmul(u0,np.linalg.inv(s0)),np.transpose(u0)))
    # print("Mat2",Mat2)
    # X = np.zeros((50,3))
    # X = Mat2 /Mat1
    # print(X)

def plotData(input_0,input_1 , sample_0,sample_1) :
    # Plot the input data and generated data
    input_0_H = (input_0[:,0]).tolist()
    input_0_W = (input_0[:,1]).tolist()
    input_1_H = (input_1[:,0]).tolist()
    input_1_W = (input_1[:,1]).tolist()
    sample_0_H = (sample_0[:, 0]).tolist()
    sample_0_W = (sample_0[:, 1]).tolist()
    sample_1_H = (sample_1[:, 0]).tolist()
    sample_1_W = (sample_1[:, 1]).tolist()

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(input_0_H,input_0_W ,c='Pink')
    ax.scatter(input_1_H,input_1_W , c = 'Blue')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title("Input Data Points")

    ax = fig.add_subplot(212)
    ax.scatter(sample_0_H, sample_0_W, c='Pink', label = 'W')
    ax.scatter(sample_1_H, sample_1_W, c='Blue', label ='M')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title("Generated Data Points")
    plt.tight_layout()
    plt.show()

def main():
    trainingSet = [[170, 57, 32, 'W'], [192, 95, 28, 'M'], [150, 45, 30, 'W'], [170, 65, 29, 'M'],
                   [175, 78, 35, 'M'], [185, 90, 32, 'M'], [170, 65, 28, 'W'], [155, 48, 31, 'W'],
                   [160, 55, 30, 'W'], [182, 80, 30, 'M'], [175, 69, 28, 'W'], [180, 80, 27, 'M'],
                   [160, 50, 31, 'W'], [175, 72, 30, 'M']]
    testSet  = [[155, 40, 35],[170, 70, 32],[175, 70, 35],[180,90,20]]
    testData = np.matrix(testSet)

    X_Women_0 , X_Men_1 = seperateClasses(trainingSet)
    mean_0 , mean_1 = calculateMean(X_Women_0 ,X_Men_1)
    phi_0 , phi_1 = calculatePrior(X_Women_0,X_Men_1 )
    sigma,sigma_0,sigma_1 = calculatesigma(X_Women_0,X_Men_1,mean_0,mean_1)
    determineClass(mean_0,mean_1,sigma_0,sigma_1,sigma,phi_0,phi_1,testData)
    sample_W, sample_M = generateFeatures(mean_0,mean_1,sigma_0,sigma_1,phi_0,phi_1)
    plotData(X_Women_0,X_Men_1,sample_W,sample_M)

main()