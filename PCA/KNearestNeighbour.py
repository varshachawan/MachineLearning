# Chawan,Varsha Rani
# 1001553524
# KNN
import math
import operator

def cartesianDistance(instance1, instance2, length):
    # calculating the cartesian distance for the height, weight and age attribute
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    squaredDist = math.sqrt(distance)
    return squaredDist

def getResult(neighbors):
    # Counting the class values for the K neighbours to decide the class with maximum count
    classCount = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classCount:
             classCount[response] += 1
        else:
            classCount[response] = 1
    sortedCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedCount[0][0]


def sortDistance(trainingSet, testInstance):
    # sorting the data based on the ascending order of distance
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = cartesianDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    return distances

def getNeighbours(distances,k):
    # getting close neighbours to the input test data based on K value
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def KNN(trainingSet,testSet,K):
    ## The starting point of the algorithm
    # Training and test data and K value
    Test_pred = []
    for row in testSet:
        dist = sortDistance(trainingSet, row[:-1])
        neighbors = getNeighbours(dist,K)
        result = getResult(neighbors)
        Test_pred.append(result)
    return Test_pred

