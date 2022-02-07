#Import Python libraries
import sys
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, classification_report
from lime.lime_tabular import LimeTabularExplainer


#Import my scripts
from utility import plot_performances



#Function to tune and fit a SVC; it returns metrics and predictions
def my_support_vector_machine(trainData, trainLabels, testData, testLabels):
    #-----------------------------------------
    #Hyperparameters' tuning (C and Kernel)  |
    #-----------------------------------------
    paramsGrid = {'C': [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'kernel': ['linear', 'poly', 'rbf']}
    grid = GridSearchCV(SVC(probability=True, random_state=42), paramsGrid, refit=True, verbose=0)
    grid.fit(trainData, trainLabels)
    print("The best found parameters are: ", grid.best_params_)

    bestC = grid.best_params_['C']            #best C
    bestKernel = grid.best_params_['kernel']  #best Kernel

    #Define the tuned model
    svc_best = SVC(C=bestC, kernel=bestKernel, probability=True, random_state=42)  #create SVC with the best hyperparams
    skf = StratifiedKFold(n_splits=10, shuffle=True)


    # --------------------------------------------
    # Fit the final model on the whole dataset  |
    # --------------------------------------------
    # "compute_evaluation_metrics_SVC":
    #   - computes, prints and returns TRAINING, and VALIDATION accuracy, precision, sensitivity, specificity
    #   - returns the final predictions on the testset, using the final model fitted on the whole dataset,
    #     with the tuned hyperparameters
    accT, precT, sensT, specT, accV, precV, sensV, specV, predictions, all_accT, all_accV = compute_evaluation_metrics_SVC(svc_best, skf, trainData, trainLabels, testData)

    #Compute accuracy, precision, sensitivity, specificity of the final fitted model
    tn, fp, fn, tp = confusion_matrix(testLabels, predictions).ravel()  #retrieve tn, fp, fn, tp
    accuracyTEST    = accuracy_score(testLabels, predictions)
    precisionTEST   = precision_score(testLabels, predictions)
    sensitivityTEST = recall_score(testLabels, predictions)
    specificityTEST = tn / (tn + fp)
    print('\nTEST RESULTS:\nAccuracy:    ', accuracyTEST, "\nPrecision:   ", precisionTEST, "\nSensitivity: ",
          sensitivityTEST, "\nSpecificity: ", specificityTEST)

    ACC = [accT, accV, accuracyTEST]
    PREC = [precT, precV, precisionTEST]
    SENS = [sensT, sensV, sensitivityTEST]
    SPEC = [specT, specV, specificityTEST]

    return svc_best, bestC, bestKernel, ACC, PREC, SENS, SPEC, predictions, all_accT, all_accV




#Function to evaluate a fitted SVC through SKFold Cross Validation
def compute_evaluation_metrics_SVC(model, skf, trainData, trainLabels, testData):
    all_accuracyTRAIN = []
    all_precisionTRAIN = []
    all_sensitivityTRAIN = []
    all_specificityTRAIN = []
    all_accuracyVAL = []
    all_precisionVAL = []
    all_sensitivityVAL = []
    all_specificityVAL = []
    print("Performing Cross-Validation...")
    for train, test in skf.split(trainData, trainLabels):  #Run the cross-validation
        #TRAINING PHASE
        model.fit(trainData[train], trainLabels[train])
        predictions = model.predict(trainData[train])
        tn, fp, fn, tp = confusion_matrix(trainLabels[train], predictions).ravel()

        all_accuracyTRAIN.append(accuracy_score(trainLabels[train], predictions))    #accuracy = (TP + TN) / (TP+TN+FP+FN)
        all_precisionTRAIN.append(precision_score(trainLabels[train], predictions))  #precision = TP / (TP + FP)
        all_sensitivityTRAIN.append(recall_score(trainLabels[train], predictions))   #sensitivity = recall = TP / (TP+FN)
        all_specificityTRAIN.append(tn / (tn + fp))  #there isn't a function for specificity

        #VALIDATION PHASE
        predictions = model.predict(trainData[test])
        tn, fp, fn, tp = confusion_matrix(trainLabels[test], predictions).ravel()

        all_accuracyVAL.append(accuracy_score(trainLabels[test], predictions))
        all_precisionVAL.append(precision_score(trainLabels[test], predictions))
        all_sensitivityVAL.append(recall_score(trainLabels[test], predictions))  #sensitivity = recall = TP / (TP+FN)
        all_specificityVAL.append(tn / (tn + fp))  #there isn't a function for specificity

    accuracyTRAINING = np.mean(all_accuracyTRAIN)
    precisionTRAINING = np.mean(all_precisionTRAIN)
    sensitivityTRAINING = np.mean(all_sensitivityTRAIN)
    specificityTRAINING = np.mean(all_specificityTRAIN)
    accuracyVALIDATION = np.mean(all_accuracyVAL)
    precisionVALIDATION = np.mean(all_precisionVAL)
    sensitivityVALIDATION = np.mean(all_sensitivityVAL)
    specificityVALIDATION = np.mean(all_specificityVAL)
    print('\nTRAINING RESULTS:\nAccuracy:   ', accuracyTRAINING, "\nPrecision:   ", precisionTRAINING, "\nSensitivity: ",
          sensitivityTRAINING, "\nSpecificity: ", specificityTRAINING)
    print('\nVALIDATION RESULTS:\nAccuracy:   ', accuracyVALIDATION, "\nPrecision:   ", precisionVALIDATION, "\nSensitivity: ",
          sensitivityVALIDATION, "\nSpecificity: ", specificityVALIDATION)

    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)

    return accuracyTRAINING, precisionTRAINING, sensitivityTRAINING, specificityTRAINING, \
            accuracyVALIDATION, precisionVALIDATION, sensitivityVALIDATION, specificityVALIDATION, \
            predictions, all_accuracyTRAIN, all_accuracyVAL





#Function to save classification report, confusion matrix, ROC curve, premutation importance
def visualize_evaluation_metrics_SVC(svc_best, ACC, PREC, SENS, SPEC, classNames, featureNames, testData, testLabels, predictions, bestC, bestKernel, saveFigures, savingPath):
    # ---------------------------------------------------------------------
    # Show confusion matrix + classification report on the TEST dataset  |
    # ---------------------------------------------------------------------
    print("\n[SVM] Classification report")

    logFile = open("images/svm/classificationReport_SVM.txt", "w+")

    # To write both in terminal and file
    class Unbuffered:
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            self.stream.write(data)
            self.stream.flush()
            logFile.write(data)  # Write the data of stdout here to a text file as well

        def flush(self):
            pass

    sys.stdout = Unbuffered(sys.stdout)
    print(classification_report(testLabels, predictions))  # show the classification report
    logFile.close()
    sys.stdout = sys.__stdout__

    confusionMatrix = confusion_matrix(testLabels, predictions)  # compute the confusion matrix

    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classNames)
    fig, ax = plt.subplots(figsize=(12, 8))
    font = {'size': 17}
    plt.rc('font', **font)
    disp.plot(ax=ax, cmap="magma")
    plt.title("Confusion matrix: C=" + str(bestC) + "; kernel=" + bestKernel, fontweight='bold')
    if saveFigures:
        plt.savefig(savingPath + "svm/confusionMatrix_SVM_best.jpg", dpi=300, bbox_inches='tight')

    # ------------------
    # Show ROC curve  |
    # ------------------
    probs = svc_best.predict_proba(testData)
    probabilityOfPositiveClass = probs[:, 1]
    fpr, tpr, threshold = roc_curve(testLabels, probabilityOfPositiveClass)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('Receiver Operating Characteristic', fontweight='bold')
    plt.plot(fpr, tpr, c='blue', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], c='red', ls='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if saveFigures:
        plt.savefig(savingPath + "svm/ROC_SVM_best.jpg", dpi=300, bbox_inches='tight')

    plot_performances(testLabels, ACC, PREC, SENS, SPEC, savingPath, "PERFORMANCES_SVM_best.jpg")


    if bestKernel == 'linear':
        fig, ax = plt.subplots(figsize=(12, 8))
        coef = svc_best.coef_[0]
        coef, names = zip(*sorted(zip(coef, featureNames)))
        plt.barh(range(len(names)), coef, align='center')
        plt.yticks(range(len(names)), names)
        plt.title("Feature importance - SVM_best (from coef. of the linear kernel)", fontweight='bold')
        plt.savefig(savingPath + "svm/FeatureImportance_SVM_best_learnKernel.jpg", dpi=300, bbox_inches='tight')

    # -----------------------------------------------------------
    # Compute permutation of feature importance on the testset |
    # -----------------------------------------------------------
    perm_importance = permutation_importance(svc_best, testData, predictions)
    sorted_idx = perm_importance.importances_mean.argsort()  # sort features by importance

    fig, ax = plt.subplots(figsize=(12, 8))
    coef, names = zip(*sorted(zip(perm_importance.importances_mean[sorted_idx], featureNames)))
    plt.barh(range(len(names)), coef, align='center')
    plt.yticks(range(len(names)), names)
    plt.title("Features permutation Importance - SVM_best", fontweight='bold')
    plt.xlabel("Permutation Importance")
    if saveFigures:
        plt.savefig(savingPath + "svm/permutationImportance_SVM_best.jpg", dpi=300, bbox_inches='tight')


#Function to retrieve SHAP exai results
def shap_svm(svm_best, testData, savingPath, featureNames):

    #Specific SHAP implementation Kernel models like SVM
    explainer = shap.KernelExplainer(svm_best.predict_proba, testData)
    shap_values = explainer.shap_values(testData)

    #Visualize global feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, testData, feature_names=featureNames, show=False, max_display=20, class_names=["NO heart disease", "Heart disease"])
    plt.savefig(savingPath + "svm/SHAP_SVM_globalFeatureImportance.jpg", dpi=300, bbox_inches='tight')

    #Visualize negative global feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values[0], testData, feature_names=featureNames, show=False, max_display=20, class_names=["NO heart disease", "Heart disease"])
    plt.savefig(savingPath + "svm/SHAP_SVM_featureImportance_negative.jpg", dpi=300, bbox_inches='tight')

    # Visualize positive global feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values[1], testData, feature_names=featureNames, show=False, max_display=20, class_names=["NO heart disease", "Heart disease"])
    plt.savefig(savingPath + "svm/SHAP_SVM_featureImportance_positive.jpg", dpi=300, bbox_inches='tight')



#Function to retrieve LIME exai results
def lime_svm(svc_best, trainData, testData, testLabels, featureNames, savingPath):
    #Specific LIME implementation for tabular data
    #in my dataset: categorical_features=SEX
    explainer = LimeTabularExplainer(trainData, feature_names=featureNames, class_names=['NO heart disease', 'Heart disease'],
                                     categorical_features=[1], verbose=True, mode='classification')

    # Calculate the explanation for one instance of the test set
    id = 9  #true instance
    exp = explainer.explain_instance(testData[id], svc_best.predict_proba, num_features=len(featureNames))

    print("--------------------------------------")
    print('[EXAI SVM] Document id: %d' % id)
    print('[EXAI SVM] Probability(Heart disease) =', svc_best.predict_proba([testData[id]])[0, 1])
    print('[EXAI SVM] True class: %s' % testLabels[id])
    # Visualize the explanation
    print(exp.as_list())
    with open("images/svm/EXAI_LIME_SVM_true.html", "w", encoding="utf-8") as file:
        file.write(exp.as_html())
    print("[EXAI SVM] Wrote results on the 'images/svm/EXAI_LIME_SVM_true.html' file")
    print("--------------------------------------")

    id = 15  #false instance
    exp = explainer.explain_instance(testData[id], svc_best.predict_proba, num_features=len(featureNames))
    print("--------------------------------------")
    print('[EXAI SVM] Document id: %d' % id)
    print('[EXAI SVM] Probability(Heart disease) =', svc_best.predict_proba([testData[id]])[0, 1])
    print('[EXAI SVM] True class: %s' % testLabels[id])
    # Visualize the explanation
    print(exp.as_list())

    with open("images/svm/EXAI_LIME_SVM_false.html", "w", encoding="utf-8") as file:
        file.write(exp.as_html())

    print("[EXAI SVM] Wrote results on the 'images/svm/EXAI_LIME_SVM_false.html' file")
    print("--------------------------------------")

    #----------------------------------------------------------------------------------

    #Analyze features' importances
    print("\n[EXAI SVM] Computing features' importances")

    importances = {i: 0.0 for i in range(testData.shape[1])}

    for i in range(testData.shape[0]):
        if i % 20 == 0:
            print("[EXAI SVM] Analyzing test data... (" + str(i) + "/" + str(testData.shape[0]) + ")")

        exp = explainer.explain_instance(testData[i], svc_best.predict_proba, num_features=len(featureNames))

        exp_map = exp.as_map()

        #get all feature labels
        feat = [exp_map[1][m][0] for m in range(len(exp_map[1]))]

        #get all feature weights of class "1"
        weight = [exp_map[1][m][1] for m in range(len(exp_map[1]))]

        #sum the weights, for each feature individually
        for m in range(len(feat)):
            importances[feat[m]] = importances[feat[m]] + np.abs(weight[m])

        #normalize the distribution
    for i in range(testData.shape[1]):
        importances[i] = importances[i] / (testData.shape[0] * 1.0)

    #Get all feature weights
    weight = [importances[m] for m in range(testData.shape[1])]

    #Visualize global feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    sorted_idx = np.array(weight).argsort()

    sorted_names = [featureNames[k] for k in sorted_idx]

    coef, names = zip(*sorted(zip(np.array(weight)[sorted_idx], sorted_names)))
    plt.barh(range(len(names)), coef, align='center')
    plt.yticks(range(len(names)), names)
    plt.title("Global feature importance - SVM_best", fontweight='bold')
    plt.savefig(savingPath + "svm/LIME_globalFeatureImportance_SVM_best.jpg", dpi=300, bbox_inches='tight')



def exai_svm(svc_best, featureNames, trainData, testData, testLabels, savingPath):

    #Run SHAP for SVM:
    shap_svm(svc_best, testData, savingPath, featureNames)

    #Run LIME for SVM:
    lime_svm(svc_best, trainData, testData, testLabels, featureNames, savingPath)

