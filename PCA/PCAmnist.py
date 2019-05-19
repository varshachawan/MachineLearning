# Chawan,Varsha Rani
# 1001553524
# PCA_ReduceDimensions_classification

import numpy as np
import glob
import scipy.misc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import KNearestNeighbour as KNNclassifier

def plot_images(data,K):
    #########################################################################
    # Plotting the K images for top K significant components
    # These images are plotted for significant K Eigen vectors formed
    #########################################################################

    for j in range(K):
        plt.subplot(5,10,j+1)
        reshaped = np.array(data[j]).reshape(28,28)
        plt.axis('off')
        plt.imshow(reshaped , cmap='gray')
    # plt.title("images for {}".format(K) +"component Eigen vectors" )
    plt.show()


def cal_Accuracy(y,yhat):
    #########################################################################
    #  Comparing the predicted values with actual to find the accuracy
    #########################################################################

    correct = 0
    for i in range(len(y)):
        if y[i] == yhat[i]:
            correct += 1
    acc =(correct / float(len(y))) * 100
    return acc


def formatData(newData,labels):
    #########################################################################
    # Formatting the reduced data from matrix to List of lists and appending
    # the corresponding label to each data
    #########################################################################

    data_array = np.array(newData)
    data_list = []
    for data, label in zip(data_array, labels):
        sub_list = []
        for i in data:
            sub_list.append(i)
        sub_list.append(label)
        data_list.append(sub_list)
    return data_list


def calc_pca(data, component):
    mean = np.mean(data, axis=0)
    #########################################################################
    #  Normalise or centre the data by subtracting Mean from original data
    #########################################################################

    centered_data = data - mean
    covariance = np.cov(centered_data, rowvar=0)
    eigen_values, eigen_vectors = np.linalg.eig(np.mat(covariance))
    #########################################################################
    # Sorting K Eigen values with high variance and retrieving the index,
    # then filter Eigen vectors corresponding to sorted K eigen values
    #########################################################################

    eigen_values_sorted = np.argsort(-eigen_values)
    eigen_values_kComp = eigen_values_sorted[:component]
    eigen_vectors_kComp = eigen_vectors[:, eigen_values_kComp]
    reduced_dim_data = centered_data * eigen_vectors_kComp
    return eigen_vectors_kComp.T.real, reduced_dim_data.real


def readImages(file):
    #########################################################################
    #  Logic to read the images , retrieve the label from filename,
    #  shuffle and split the data
    #########################################################################

    data_list = []
    lables = []
    directory = file + "/*.png"

    for fname in glob.glob(directory):
        img = scipy.misc.imread(fname).astype(np.float32)
        img = img.flatten()
        # img = img / 127.5
        # img = img - 1
        data_list.append(img)

        #For windows  OS based \\
        key, value = fname.split('\\')
        tmp = int(value[0])
        lables.append(tmp)

    data_list = np.array(data_list)

    data_list, lables = shuffle(data_list, lables)
    # return data_list , lables
    X_train, X_test, Y_train, Y_test = train_test_split(data_list, lables)
    return X_train, X_test, Y_train, Y_test


def Main():

    X_train, X_test, Y_train, Y_test = readImages("Train_Data_4")
    #25 good 100- 57 better 150, 52.47 2-65 5-68 -87.
    # X_train, Y_train = readImages("Train_data")
    # X_test, Y_test = readImages("Test_data")
    K_components =[2,5,10,20,50]

    #########################################################################
    #  PCA Algorithm for various K components
    #########################################################################

    for component in K_components :
        eigen_vectors_T, red_dim_train= calc_pca(X_train, component)
        _,red_dim_test = calc_pca(X_test,component)
        plot_images(eigen_vectors_T, component)
        print("================================== For most significant ",component,"Components ===========================================")
        print("The Input dimensions for train data:",np.array(X_train).shape)
        print("The reduced dimensions for train data:" ,red_dim_train.shape)

        train_data = formatData(red_dim_train,Y_train)
        test_data = formatData(red_dim_test,Y_test)

        #########################################################################
        #  Classification using KNN
        #########################################################################

        print("############ K Nearest Neighbour ############")

        for K in [5,10,15,19]:
            pred_test = KNNclassifier.KNN(train_data,test_data,K)
            accuracy = cal_Accuracy(pred_test,Y_test)
            print("For K =",K,"accuracy:",accuracy)

        #########################################################################
        #  Classification using Logistic Regression
        #########################################################################

        Y = np.array(Y_train)
        clf = LogisticRegression(max_iter= 5000,solver='lbfgs',multi_class='multinomial')
        clf.fit(red_dim_train,Y)
        Y_pred_Log = clf.predict(red_dim_test)
        accuracy = cal_Accuracy(Y_pred_Log, Y_test)
        print("######### L0GISTIC REGRESSION ###########")
        print("accuracy:",accuracy)

        #########################################################################
        #  Classification using Random Forest
        #########################################################################

        classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
        classifier.fit(red_dim_train, Y)
        Y_pred_RandForest = classifier.predict(red_dim_test)
        accuracy = cal_Accuracy(Y_pred_RandForest, Y_test)
        print("######### RANDOM FOREST ###########")
        print("accuracy:", accuracy)


Main()