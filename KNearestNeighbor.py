# Chawan,Varsha Rani
# 1001553524
import math
import operator
import sys

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


def main():
    ## Training and test data and K value

    trainingSet = [[170, 57, 32, 'W'],[192, 95, 28, 'M'],[150, 45, 30, 'W'],[170, 65, 29, 'M'],
                   [175, 78, 35, 'M'],[185, 90, 32, 'M'],[170, 65, 28, 'W'],[155, 48, 31, 'W'],
                   [160, 55, 30, 'W'],[182, 80, 30, 'M'],[175, 69, 28, 'W'],[180, 80, 27, 'M'],
                   [160, 50, 31, 'W'],[175, 72, 30, 'M']]
    # trainingSet = [[170, 57, 'W'],[192, 95,'M'],[150, 45,  'W'],[170, 65,'M'],
    #            [175, 78,'M'],[185, 90,'M'],[170, 65,'W'],[155, 48,'W'],
    #            [160, 55,'W'],[182, 80,'M'],[175, 69,'W'],[180, 80,'M'],
    #            [160, 50,'W'],[175, 72,'M']]
    K = int(sys.argv[1])
    test = (sys.argv[2:])
    testdata = []
    for v in test:
        testdata.append(int(v))
    testSet = [testdata]
    dist = sortDistance(trainingSet, testSet[0])
    neighbors = getNeighbours(dist,K)
    result = getResult(neighbors)
    print("The Predicted Gender is ",result)

main()