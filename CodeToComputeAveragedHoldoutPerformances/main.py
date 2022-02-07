#Import Python libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns


#Import my scripts
from randomForest import my_random_forest, visualize_evaluation_metrics_RF, exai_rf
from svm import my_support_vector_machine, visualize_evaluation_metrics_SVC, exai_svm
from utility import histogram_features_and_labels, plot_accuracies_train_val_test, plot_performances

savingPath = "images/"  #set the output plots path

#Import the dataset from 'processed.cleveland.data' file
featureNames = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholesterol', 'fasting blood pressure',
                'resting electrocardiographic results', 'maximum heart rate achieved', 'exercise induced angina',
                'ST depression induced by exercise relative to rest', 'slope of the peak exercise ST segment',
                'number of major vessels colored by flourosopy', 'thalassemia']

classNames = ["NO Heart disease", "Heart disease"]
datasetCols = featureNames + ['goal']

dataset = pd.read_csv('processed.cleveland.data', header=None, names=datasetCols)  #no header present

#---- If some features have to be removed: -----#
#dataset = dataset.drop(labels=['thalassemia', 'number of major vessels colored by flourosopy', 'chest pain type'], axis=1)
#featureNames.remove('thalassemia')
#featureNames.remove('number of major vessels colored by flourosopy')
#featureNames.remove('chest pain type')

print(featureNames)
print(dataset.head)  #to have a preview of the dataframe



#Retrieve true labels: according to the specs, if 'goal' is greater than 0, a heart disease is present, so the
#                      classification labels are simplified as True=Heart Disease; False=NO Heart Disease
labels = np.array(dataset['goal'] > 0)
#print(np.count_nonzero(labels), len(labels)-np.count_nonzero(labels))  #to check the class balance (139 vs 164)

#---------------------------------------------------------------------
# Retrieve data (as string, because of the needed "data cleaning"!)  |
#---------------------------------------------------------------------
data = np.array(dataset.iloc[:, 0:len(featureNames)]).astype(str)
print(data.shape)  #to check

#There are 6 '?' fields inside the dataset. A proposed solution to avoid removing that row is to substitute that random
#character with the most common value of that category
indices = np.where(np.char.find(data, "?") == 0)
#print(indices)  #to check

for k in range(indices[0].size):
    row = indices[0][k]
    col = indices[1][k]
    #print(row, col)  #to check
    unique, pos = np.unique(data[:, col], return_inverse=True)  #to find all unique elements and their positions
    counts = np.bincount(pos)  #to count the number of each unique element
    maxpos = counts.argmax()  #to find the positions of the maximum count

    #print("Before: ", data[row, col])  #to check
    data[row, col] = unique[maxpos]
    #print("After: ", data[row, col])  #to check



#-----------------------------
#Data scaling and transform  |
#-----------------------------
scaler = MinMaxScaler()  #StandardScaler()
new_data = scaler.fit_transform(data).astype(float)





numberOfPartitions = 20
meanACC_SVM = np.empty([1,3])   #accT, accV, accuracyTEST
meanPREC_SVM = np.empty([1,3])  #...
meanSENS_SVM = np.empty([1,3])  #...
meanSPEC_SVM = np.empty([1,3])  #...
meanACC_RF = np.empty([1,3])   #accT, accV, accuracyTEST
meanPREC_RF = np.empty([1,3])  #...
meanSENS_RF = np.empty([1,3])  #...
meanSPEC_RF = np.empty([1,3])  #...

for i in range(numberOfPartitions):
    print("[SVM] Considering partition...", i)
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

    #--------------------------------------------------
    # RANDOMLY Split data and labels into train/test  |
    #--------------------------------------------------
    trainData, testData, trainLabels, testLabels = train_test_split(new_data, labels, test_size=0.2, shuffle=True)
    #To check the number of samples: shuffle=False because of the difference in performances
    #print(np.count_nonzero(testLabels), len(testLabels)-np.count_nonzero(testLabels), np.count_nonzero(trainLabels), len(trainLabels)-np.count_nonzero(trainLabels))


    #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*  Support Vector Machine  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
    #----------------------------------------------------------------------------------------
    # Call the SVM script, get best model, metrics and test predictions of the tuned model  |
    # ---------------------------------------------------------------------------------------
    svc_best, bestC, bestKernel, ACC, PREC, SENS, SPEC, predictions, all_accT, all_accV = my_support_vector_machine(trainData, trainLabels, testData, testLabels)

    #Store current values to then compute the mean
    meanACC_SVM += np.array(ACC)
    meanPREC_SVM += np.array(PREC)
    meanSENS_SVM += np.array(SENS)
    meanSPEC_SVM += np.array(SPEC)
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

    print("[RF] Considering partition...", i)
    
    #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*  Random Forest  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
    #--------------------------------------------------------------------------------------------------
    # Call the Random Forest script, get best model, metrics and test predictions of the tuned model  |
    #--------------------------------------------------------------------------------------------------
    rf_best, bestMaxDepth, bestMaxFeatures, bestMaxSamples, ACC, PREC, SENS, SPEC, predictions, all_accT, all_accV = my_random_forest(trainData, trainLabels, testData, testLabels)

    #Store current values to then compute the mean
    meanACC_RF += np.array(ACC)
    meanPREC_RF += np.array(PREC)
    meanSENS_RF += np.array(SENS)
    meanSPEC_RF += np.array(SPEC)
    print(meanACC_RF)

    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

#Compute the mean performances over the "numberOfPartitions" iterations
meanACC_SVM = meanACC_SVM / numberOfPartitions
meanPREC_SVM = meanPREC_SVM / numberOfPartitions
meanSENS_SVM = meanSENS_SVM / numberOfPartitions
meanSPEC_SVM = meanSPEC_SVM / numberOfPartitions

meanACC_RF = meanACC_RF / numberOfPartitions
meanPREC_RF = meanPREC_RF / numberOfPartitions
meanSENS_RF = meanSENS_RF / numberOfPartitions
meanSPEC_RF = meanSPEC_RF / numberOfPartitions

# ---------------------
# Performances plots  |
# ---------------------
plot_performances(testLabels, meanACC_SVM[0].tolist(), meanPREC_SVM[0].tolist(), meanSENS_SVM[0].tolist(), meanSPEC_SVM[0].tolist(), savingPath, "PERFORMANCES_SVM_best.jpg")
plot_performances(testLabels, meanACC_RF[0].tolist(), meanPREC_RF[0].tolist(), meanSENS_RF[0].tolist(), meanSPEC_RF[0].tolist(), savingPath, "PERFORMANCES_RF_best.jpg")




