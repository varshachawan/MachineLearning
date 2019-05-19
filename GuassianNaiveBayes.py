# Chawan,Varsha Rani
# 1001553524
import math
import sys

def calculateMeanVariance(numbers):
    # calculate Mean and Variance for every Feature Per class
    mean = sum(numbers) / float(len(numbers))
    # avg = mean(numbers)
    variances = sum([pow(x - mean, 2) for x in numbers]) / float(len(numbers) - 1)
    return mean, variances


def probabilityCalculations(x, mean, variance):
    # calculate the probabilities per feature
    prob = (1 / (math.sqrt(2 * math.pi * variance))) * (math.exp(-(math.pow(x - mean, 2) / (2 * variance))))
    return prob

def aggregateClass(dataset):
    # Logic to aggregate the features based on the classes
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
        del vector[-1]
    return separated


def meanVariance(dataset):
    # overall summary of Mean and variances for every feature
    summaries = [(calculateMeanVariance(attribute)) for attribute in zip(*dataset)]
    return summaries

def meanVarianceperClass(dataset):
    # mean and variances segregated per class
    separated = aggregateClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = meanVariance(instances)
    return summaries


def calculateClassProbabilities(summaries, inputVector):
    # calculate the posterior probabilities
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, variance = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= probabilityCalculations(x, mean, variance)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def main():
    # Training and test data

    trainingSet = [[170, 57, 32, 'W'],[192, 95, 28, 'M'],[150, 45, 30, 'W'],[170, 65, 29, 'M'],
                   [175, 78, 35, 'M'],[185, 90, 32, 'M'],[170, 65, 28, 'W'],[155, 48, 31, 'W'],
                   [160, 55, 30, 'W'],[182, 80, 30, 'M'],[175, 69, 28, 'W'],[180, 80, 27, 'M'],
                   [160, 50, 31, 'W'],[175, 72, 30, 'M']]
    # trainingSet = [[170, 57, 'W'],[192, 95,'M'],[150, 45,  'W'],[170, 65,'M'],
    #            [175, 78,'M'],[185, 90,'M'],[170, 65,'W'],[155, 48,'W'],
    #            [160, 55,'W'],[182, 80,'M'],[175, 69,'W'],[180, 80,'M'],
    #            [160, 50,'W'],[175, 72,'M']]
    test = (sys.argv[1:])
    testdata = []
    for v in test:
        testdata.append(int(v))
    testSet = [testdata]
    summaries = meanVarianceperClass(trainingSet)
    result = calculateClassProbabilities(summaries, testSet[0])
    print("The predicted class is ",result)

main()