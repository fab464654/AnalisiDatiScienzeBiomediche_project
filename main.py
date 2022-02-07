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
from utility import histogram_features_and_labels, plot_accuracies_train_val_test

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

#---------------------------------------------
# Visualize data (feature values and labels) |
#---------------------------------------------
#histogram_features_and_labels(dataset, datasetCols, savingPath)


#-----------------------------
#Data scaling and transform  |
#-----------------------------
scaler = MinMaxScaler()  #StandardScaler()
new_data = scaler.fit_transform(data).astype(float)
#new_data = data  #uncomment to skip scaling


#------------------------------
# Feature correlation heatmap |
#------------------------------
featuresData = pd.DataFrame(data=new_data, index=None, columns=featureNames)  #need to create a DataFrame to create heatmap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(featuresData.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
plt.savefig('images/correlationMatrix.png', dpi=300, bbox_inches='tight')



#----------------------------------------
#Split data and labels into train/test  |
#----------------------------------------
trainData, testData, trainLabels, testLabels = train_test_split(new_data, labels, test_size=0.2, shuffle=False)
#To check the number of samples: shuffle=False because of the difference in performances
#print(np.count_nonzero(testLabels), len(testLabels)-np.count_nonzero(testLabels), np.count_nonzero(trainLabels), len(trainLabels)-np.count_nonzero(trainLabels))


#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*  Support Vector Machine  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
#----------------------------------------------------------------------------------------
# Call the SVM script, get best model, metrics and test predictions of the tuned model  |
# ---------------------------------------------------------------------------------------
svc_best, bestC, bestKernel, ACC, PREC, SENS, SPEC, predictions, all_accT, all_accV = my_support_vector_machine(trainData, trainLabels, testData, testLabels)

plot_accuracies_train_val_test(all_accT, all_accV, ACC[2], "images/svm/checkOverfit.jpg")

#-----------------------------------------------------------------------------------------------------------------------------
#Compute, visualize and save all evaluation metrics (classification report, confusion matrix, ROC curve, performances plot)  |
#-----------------------------------------------------------------------------------------------------------------------------
visualize_evaluation_metrics_SVC(svc_best, ACC, PREC, SENS, SPEC, classNames, featureNames, testData, testLabels, predictions, bestC, bestKernel, saveFigures=True, savingPath=savingPath)


#----------------------------------------
# EXplainable AI for SVM (lime + shap)  |
#----------------------------------------
exai_svm(svc_best, featureNames, trainData, testData, testLabels, savingPath)




#-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*  Random Forest  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
#--------------------------------------------------------------------------------------------------
# Call the Random Forest script, get best model, metrics and test predictions of the tuned model  |
#--------------------------------------------------------------------------------------------------
rf_best, bestMaxDepth, bestMaxFeatures, bestMaxSamples, ACC, PREC, SENS, SPEC, predictions, all_accT, all_accV = my_random_forest(trainData, trainLabels, testData, testLabels)

plot_accuracies_train_val_test(all_accT, all_accV, ACC[2], "images/rf/checkOverfit.jpg")


#-----------------------------------------------------------------------------------------------------------------------------
# Compute, visualize and save all evaluation metrics (classification report, confusion matrix, ROC curve, performances plot  |
#-----------------------------------------------------------------------------------------------------------------------------
visualize_evaluation_metrics_RF(rf_best, ACC, PREC, SENS, SPEC, classNames, featureNames, testData, testLabels, predictions, bestMaxDepth, bestMaxFeatures, bestMaxDepth, saveFigures=True, savingPath=savingPath)

#--------------------------------------------
# EXplainable AI for RF (lime + shap)
#--------------------------------------------
exai_rf(rf_best, featureNames, trainData, testData, testLabels, savingPath)


#Visualizing the decision tree from the rf_best trained model
#Reference: https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
from sklearn.tree import export_graphviz
from subprocess import call

#Extract single tree
estimator = rf_best.estimators_[5]

#Export as dot file
export_graphviz(estimator, out_file='images/rf/tree.dot', feature_names=featureNames, class_names=classNames, rounded=True,
                proportion=False, precision=2, filled=True)

#Convert to png using system command (requires Graphviz)
call(['dot', '-Tpng', 'images/rf/tree.dot', '-o', 'images/rf/rf_decisionTree.png', '-Gdpi=600'])




