# Chawan,Varsha Rani
# 1001553524
# Semi supervised
import LogisticReg as model

def pred_unlabbled(unlabelledSet,thetha):
    prob_dataset = []
    for row in unlabelledSet :

        res,sig = model.Test(row,thetha)
        if res == 0:
            sig = 1-sig

        row.append(res)
        row.append(sig)
        prob_dataset.append(row)
        sorted_unlabelled = sorted(prob_dataset, key=lambda x: x[-1],reverse= True)
    return sorted_unlabelled

def main():
    trainingSet =[[170, 57, 32, 0], [190, 95, 28, 1], [150, 45, 35, 0], [168, 65, 29, 1],
                [175, 78, 26, 1], [185, 90, 32, 1], [171, 65, 28, 0], [155, 48, 31, 0], [165, 60, 27, 0]]

    unlabelledSet = [[182, 80, 30], [175, 69, 28], [178, 80, 27],
                    [160, 50, 31], [170, 72, 30], [152, 45, 29],
                    [177, 79, 28], [171, 62, 27], [185, 90, 30],
                    [181, 83, 28], [168, 59, 24], [158, 45, 28],
                    [178, 82, 28], [165, 55, 30], [162, 58, 28],
                    [180, 80, 29], [173, 75, 28], [172, 65, 27],
                    [160, 51, 29], [178, 77, 28], [182, 84, 27],
                    [175, 67, 28], [163, 50, 27], [177, 80, 30],
                    [170, 65, 28] ]

    testSet =  [[169, 58, 30, 0],[185, 90, 29, 1],[148, 40, 31, 0],[177, 80, 29, 1],
                [170, 62, 27, 0],[172, 72, 30, 1],[175, 68, 27, 0],[178, 80, 29, 1]]


    # model.normalize_dataset(trainingSet)
    # model.normalize_dataset(unlabelledSet)
    # model.normalize_dataset(testSet)

    l_rate = 0.001
    n_epoch = 100
    thetha_base = model.coefficients_sgd(trainingSet,l_rate,n_epoch)
    thetha = thetha_base

    k = 5
    N = len(unlabelledSet)
    num_iterations = int(N / k)
    # print("num iterations",num_iterations)

    for i in range(num_iterations):
        remaining_UL = []
        sorted_unlabelled = pred_unlabbled(unlabelledSet,thetha)

        for j in range(k):
            trainingSet.append(sorted_unlabelled[j][:4])

        if len(sorted_unlabelled)>=5:
            # print("len sorted unlabbed", len(sorted_unlabelled))
            for r in range(5,len(sorted_unlabelled)):
                remaining_UL.append(sorted_unlabelled[r][:3])
            unlabelledSet = remaining_UL
        thetha = model.coefficients_sgd(trainingSet, l_rate, n_epoch)

    print("******************** (2a)************************")
    print("The thetha(coefficients) values for classifier learned using Labelled data Ds are :", thetha_base)
    # print("len of train before sem", len(trainingSet))
    thetha_sem = model.coefficients_sgd(trainingSet, l_rate, n_epoch)

    # logic to predict the class for Test data

    # print("The predicted labels for unlabelled dataset Du are :",label_pred)


    print("The thetha(coefficients) values for Semi supervised model are: ", thetha_sem)
    #  # testing
    actual , Base_model_pred ,res_base = model.testModel(testSet,thetha_base)
    Actual = []
    for i in actual:
        if i == 0:
            Actual.append('W')
        else:
            Actual.append('M')


    actual,SelfLearned_pred, res_SL = model.testModel(testSet,thetha_sem)
    print("******************** (2b)***********************")
    print("The Actual labels for Test dataset Dt are :                                              ",Actual)
    print("The predicted labels for Test data, trained using labelled data Ds (i.e Base Model)are:  ",Base_model_pred)
    print("The predicted labels for Test Data , trained using Semi Supervised Model are :           ",SelfLearned_pred)



    accuracy_base = model.Accuracy(actual,res_base)
    accuracy_SL = model.Accuracy(actual,res_SL)
    print("Accuracy of classifier learned only from the labeled data Ds is: ",accuracy_base)
    print("Accuracy of classifier learned from  Semi Supervised Model is :", accuracy_SL)

main()