#1001553524
#Varsha Rani Chawan
#decision tree
import numpy as np
import csv


def calc_entropy_feature(data, classes,feature):
    # calculate the total entropy of the feature
    gain = 0
    nData = len(data)

    # collects all the unique values of  a feature
    feature_vals = {}
    for row in data:
        if row[feature] not in feature_vals.keys():
            feature_vals[row[feature]] = 1
        else:
            feature_vals[row[feature]] += 1

    for fi in feature_vals.keys():
        fi_entropy = 0
        row_indx = 0
        newClasses = {}
        classCounts = 0
        for row in data:
            if row[feature] == fi:
                classCounts += 1
                if classes[row_indx] in newClasses.keys():
                    newClasses[classes[row_indx]] += 1
                else:
                    newClasses[classes[row_indx]] = 1
            row_indx += 1

        for C in newClasses.keys():
            p = float(newClasses[C]) / classCounts
            fi_entropy += -p * np.log2(p)
        gain += float(feature_vals[fi]) / nData * fi_entropy
    return gain


def calc_total_entropy(classes):
    # calculates the total entropy of classes
    unique_classes = {}
    n = len(classes)
    entropy = 0
    for c in classes:
        if c not in unique_classes:
            unique_classes[c] = 1
        else:
            unique_classes[c] += 1
    for unique in unique_classes:
        prob = unique_classes[unique] / float(n)
        entropy += -prob * np.log2(prob)

    return entropy

def sub_data(data, targets, feature, fi):
    # builds data for a branch
    new_data = []
    new_targets = []
    row_idx = 0
    for row in data:
        if row[feature] == fi:
            new_row = row[:feature]
            new_row.extend(row[feature + 1:])

            new_data.append(new_row)
            new_targets.append(targets[row_idx])
        row_idx += 1
    return new_targets, new_data

def build_tree(data,classes,features,highest):
    fi_vals = np.unique(np.transpose(data)[highest])
    feature = features[highest]  # Feature Name at that "best" position
    del features[highest]
    tree = {feature: {}}
    for fi in fi_vals:
        t, d = sub_data(data, classes, highest, fi)
        # iterate the process
        subtree = create_tree(d, t, features)
        tree[feature][fi] = subtree
    return tree


def create_tree(data, classes, features):
    nFeatures = len(features)
    if len(np.unique(classes)) == 1:
        return classes[0]
    else:
        # calculates the  information gain
        totalEntropy = calc_total_entropy(classes)
        gain = np.zeros(nFeatures)
        for feature in range(nFeatures):
            feature_e = calc_entropy_feature(data, classes, feature)
            gain[feature] = totalEntropy - feature_e
        highest = np.argmax(gain)  # index of the best feature

        tree = build_tree(data,classes,features,highest)

        return tree


def tree_output(tree, data_row, features):
    if type(tree) is not dict:
        return str(tree)
    else:
        for key in tree.keys():
            f_idx = features.index(key)
            f_i = data_row[f_idx]
            return tree_output(tree[key][f_i], data_row, features)


def predict(filename, tree):
    #predicts the class for new data
    data, features ,targets = read_CSV(filename)
    correct = 0
    indx = 0
    for row in data:
        output_indx = tree_output(tree, row, features)
        if output_indx == targets[indx]:
            correct += 1
        indx += 1
    accuracy = (correct / float(len(data)))*100
    return accuracy


def read_CSV(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    features = data[0]
    del features[0]
    del data[0]
    targets = []
    for row in data:
        targets.append(row[0])
        del row[0]

    return data, features,targets

def split(data, split_fraction=20):
    split_number = int((split_fraction * len(data)) / 100)
    return split_number

def main():
    train_file = "MushroomTrain.csv"
    test_file = "MushroomTest.csv"
    data, features, classes = read_CSV(train_file)
    tree = create_tree(data, classes, features)
    Accuracy_Test = predict(test_file, tree)
    Accuracy_Train = predict(train_file, tree)
    print("============================")
    print("tree",tree)
    print("============================")
    print(" Accuracy Test Data:", Accuracy_Test)
    print("============================")
    print(" Accuracy Train Data:", Accuracy_Train)
    print("============================")

main()