# Chawan,Varsha Rani
# 1001553524
# Single Linkage Clustering
import numpy as np
import sys
import math

def cartesianDistance(instance1, instance2, length):
    # calculating the cartesian distance for the height, weight and age attribute
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    squaredDist = math.sqrt(distance)
    return squaredDist

def distArray(trainingSet):
    reall = []
    for record1 in trainingSet:
        res1 = []
        for record2 in trainingSet:
            res = cartesianDistance(record1,record2, 3)
            res1.append(res)
        reall.append(res1)
    reallarray = np.array(reall)
    return reallarray

def create_clusters(input):
    clusters = {}
    row_index = -1
    col_index = -1
    array = []

    for n in range(input.shape[0]):
        array.append(n)

    clusters[0] = array.copy()

    # finding minimum value from the distance matrix
    for k in range(1, input.shape[0]):
        min_val = sys.maxsize

        for i in range(0, input.shape[0]):
            for j in range(0, input.shape[1]):
                if (input[i][j] <= min_val):
                    min_val = input[i][j]
                    row_index = i
                    col_index = j
        print("Distance(linkage) value between the merged ses at each merge set: ",min_val)


        for i in range(0, input.shape[0]):
            if (i != col_index):
                temp = min(input[col_index][i], input[row_index][i])
                input[col_index][i] = temp
                input[i][col_index] = temp

        for i in range(0, input.shape[0]):
            input[row_index][i] = sys.maxsize
            input[i][row_index] = sys.maxsize

        # creating  dictionary of clusters


        minimum = min(row_index, col_index)
        maximum = max(row_index, col_index)
        for n in range(len(array)):
            if (array[n] == maximum):
                array[n] = minimum
        clusters[k] = array.copy()
        print(k, " orderCluster : ",clusters[k])

    return clusters
#


def main():
    trainingSet = [[170, 57, 32], [192, 95, 28], [150, 45, 30], [170, 65, 29],
                   [175, 78, 35], [185, 90, 32], [170, 65, 28], [155, 48, 31],
                   [160, 55, 30], [182, 80, 30], [175, 69, 28], [180, 80, 27],
                   [160, 50, 31], [175, 72, 30]]

    distances_matix = distArray(trainingSet)
    np.fill_diagonal(distances_matix, sys.maxsize)
    print("============================= 1(a) Single Linkage==============================")

    clusters = create_clusters(distances_matix)

main()