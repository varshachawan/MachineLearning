# Chawan,Varsha Rani
# 1001553524
# Linear Regression
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys

def createMatrix(A,B,n):
# create matrices for N order polynomial

    rows = np.size(A)
    col = int((n + 1) * (n + 2) / 2)
    X = np.zeros((rows,col))
    if n == 1:
        X[:, 0] = 1
        X[:, 1] = A
        X[:, 2] = B
    if n== 2 :
        X[:, 0] = 1
        X[:, 1] = A
        X[:, 2] = B
        X[:, 3] = A * B
        X[:, 4] = A * A
        X[:, 5] = B * B
    if n == 3 :
        X[:, 0] = 1
        X[:, 1] = A
        X[:, 2] = B
        X[:, 3] = A * B
        X[:, 4] = A * A
        X[:, 5] = B * B
        X[:, 6] = A * A * B
        X[:, 7] = A * B * B
        X[:, 8] = A * A * A
        X[:, 9] = B * B * B
    if n == 4 :
        X[:, 0] = 1
        X[:, 1] = A
        X[:, 2] = B
        X[:, 3] = A * B
        X[:, 4] = A * A
        X[:, 5] = B * B
        X[:, 6] = A * A * B
        X[:, 7] = A * B * B
        X[:, 8] = A * A * A
        X[:, 9] = B * B * B
        X[:, 10] = A * A * A * B
        X[:, 11] = A * B * B * B
        X[:, 12] = A * A * B * B
        X[:, 13] = A * A * A * A
        X[:, 14] = B * B * B * B
    return X

def calculate_Coefficients(X,Y) :
    # calculate the coeff using analytical method
    Xtranp = np.transpose(X)
    thetha = np.matmul(np.matmul(np.linalg.pinv(np.matmul(Xtranp, X)), Xtranp), Y)
    return thetha

def predict_values(n,thetha,a,b):
    # given input data predicts the output values
    if n == 1:
        predicted_Y = thetha[0] + a * thetha[1] + b * thetha[2]

    if n == 2 :
        predicted_Y = thetha[0] + a * thetha[1] + b * thetha[2] + (a * b) * thetha[3] + (a * a) * thetha[4] + (b * b) *thetha[5]

    if n == 3 :
        predicted_Y = thetha[0] + a * thetha[1] + b * thetha[2] + (a * b) * thetha[3] + (a * a) * thetha[4] + (b * b) * thetha[5]\
                  + (a * a * b) * thetha[6] + (a * b * b) * thetha[7] + (a * a * a) * thetha[8] + (b * b * b) * thetha[9]
    if n == 4:
        predicted_Y = thetha[0] + a * thetha[1] + b * thetha[2] + (a * b) * thetha[3] + (a * a) * thetha[4] + (b * b) * thetha[5] \
                      + (a * a * b) * thetha[6] + (a * b * b) * thetha[7] + (a * a * a) * thetha[8] + (b * b * b) * thetha[9]\
                      + (a * a * a * b) * thetha[10] + (a * b * b * b) * thetha[11] + (a * a * b * b) * thetha[12]\
                      + (a * a * a * a) * thetha[13] + (b * b * b * b) * thetha[14]
    return predicted_Y

def mean_square_error(ypred, Y):
    # calculate the mean square error for test data
    TSE = 0
    rows = len(Y)
    for i in range(rows):
        squared_error = (Y[i] - ypred[i]) ** 2
        TSE += squared_error
    print("The total error for test data is :", TSE)
    MSE = TSE * (1 / (2 * rows))
    print("The mean square error for test data is",MSE)
    return MSE


def plotFunction(thetha,A,B,Y,n):
    #Plot the input data and the Nth order curve fitting the data
    x_surf, y_surf = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
    a = x_surf.ravel()
    b = y_surf.ravel()
    predicted_Y = predict_values(n,thetha,a,b)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(A, B, Y, c='blue', marker='o', alpha=1.0)
    ax.plot_surface(x_surf, y_surf, predicted_Y.reshape(x_surf.shape), rstride=1, cstride=1, cmap=cm.jet, alpha=0.5)
    ax.set_xlabel('Input1')
    ax.set_ylabel('Input2')
    ax.set_zlabel('Output')
    plt.show()

def main():
    trainingData = np.array([[6.4432, 9.6309, 50.9155], [3.7861, 5.4681, 29.9852],
                             [8.1158, 5.2114, 42.9626], [5.3283, 2.3159, 24.7445],
                             [3.5073, 4.8890, 27.3704], [9.3900, 6.2406, 51.1350],
                             [8.7594, 6.7914, 50.5774], [5.5016, 3.9552, 30.5206],
                             [6.2248, 3.6744, 31.7380], [5.8704, 9.8798, 49.6374],
                             [2.0774, 0.3774, 10.0634], [3.0125, 8.8517, 38.0517],
                             [4.7092, 9.1329, 43.5320], [2.3049, 7.9618, 33.2198],
                             [8.4431, 0.9871, 31.1220], [1.9476, 2.6187, 16.2934],
                             [2.2592, 3.3536, 19.3899], [1.7071, 6.7973, 28.4807],
                             [2.2766, 1.3655, 13.6945], [4.3570, 7.2123, 36.9220],
                             [3.1110, 1.0676, 14.9160], [9.2338, 6.5376, 51.2371],
                             [4.3021, 4.9417, 29.8112], [1.8482, 7.7905, 32.0336],
                             [9.0488, 7.1504, 52.5188], [9.7975, 9.0372, 61.6658],
                             [4.3887, 8.9092, 42.2733], [1.1112, 3.3416, 16.5052],
                             [2.5806, 6.9875, 31.3369], [4.0872, 1.9781, 19.9475],
                             [5.9490, 0.3054, 20.4239], [2.6221, 7.4407, 32.6062],
                             [6.0284, 5.0002, 35.1676], [7.1122, 4.7992, 38.2211],
                             [2.2175, 9.0472, 36.4109], [1.1742, 6.0987, 25.0108],
                             [2.9668, 6.1767, 29.8861], [3.1878, 8.5944, 37.9213],
                             [4.2417, 8.0549, 38.8327], [5.0786, 5.7672, 34.4707]])


    testData = np.array([[0.8552, 1.8292, 11.5848], [2.6248, 2.3993, 17.6138],
                        [8.0101, 8.8651, 54.1331], [0.2922, 0.2867, 5.7326],
                        [9.2885, 4.8990, 46.3750], [7.3033, 1.6793, 29.4356],
                        [4.8861, 9.7868, 46.4227], [5.7853, 7.1269, 40.7433],
                        [2.3728, 5.0047, 24.6220], [4.5885, 4.7109, 29.7602]])
    print("Please provide the N value")
    N = int(sys.argv[1])
    print("For",N,"order polynomial")
    Feature1 = trainingData[:, 0]
    Feature2 = trainingData[:, 1]
    OutPutMat = trainingData[:, 2]
    X1Test = testData[:, 0]
    X2Test = testData[:, 1]
    YTest = testData[:, 2]
    InputMat = createMatrix(Feature1,Feature2,N)
    Coeff = calculate_Coefficients(InputMat,OutPutMat)
    print("The Thetha values are :", Coeff)
    Ypred = predict_values(N,Coeff,X1Test,X2Test)
    print("******************** (3c)*************************")
    print("The predicted values for test data are :",Ypred)
    error = mean_square_error(Ypred,YTest)
    plotFunction(Coeff, Feature1, Feature2, OutPutMat, N)


main()
