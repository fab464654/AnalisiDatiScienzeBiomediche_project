#Import Python libraries
import numpy as np
from matplotlib import pyplot as plt
from textwrap import wrap


def plot_performances(labels, ACC, PREC, SENS, SPEC, savingPath, savingName):
    x = np.arange(len(labels))
    barWidth = 0.2
    fig, ax = plt.subplots(figsize=(12, 8))

    #Set position of bar on X axis
    br1 = np.arange(len(ACC))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    #Make the plot
    plt.bar(br1, ACC, color=(0.533, 0.67, 0.81), width=barWidth, edgecolor='grey', label="Accuracy")
    ax.tick_params(axis='x', labelsize=18)
    for i, x_loc in enumerate(br1):
        plt.text(x_loc - 0.05, ACC[i], "{:.2f}".format(ACC[i]), fontsize=11)

    plt.bar(br2, PREC, color=(0.95, 0.627, 0.34), width=barWidth, edgecolor='grey', label="Precision")
    ax.tick_params(axis='x', labelsize=18)

    for i, x_loc in enumerate(br2):
        plt.text(x_loc - 0.05, PREC[i], "{:.2f}".format(PREC[i]), fontsize=11)

    plt.bar(br3, SENS, color=(0.525, 0.7, 0.498), width=barWidth, edgecolor='grey', label="Sensitivity")
    ax.tick_params(axis='x', labelsize=18)
    for i, x_loc in enumerate(br3):
        plt.text(x_loc - 0.05, SENS[i], "{:.2f}".format(SENS[i]), fontsize=11)

    plt.bar(br4, SPEC, color=(0.847, 0.562, 0.9), width=barWidth, edgecolor='grey', label="Specificity")

    ax.tick_params(axis='x', labelsize=18)
    for i, x_loc in enumerate(br4):
        plt.text(x_loc - 0.05, SPEC[i], "{:.2f}".format(SPEC[i]), fontsize=11)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=4, fancybox=True, shadow=True)
    plt.title("Performance metrics for each phase", fontweight='bold', fontsize=18, y=1.04)
    plt.ylabel("Score", fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth + 0.1 for r in range(3)], ["Training results", "Validation results", "Test results"])
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)

    plt.tight_layout()
    plt.savefig(savingPath + savingName, dpi=300, bbox_inches='tight')


#Function to visualize on 13 histograms the features' values and distributions
def histogram_features_and_labels(dataset, datasetCols, savingPath):
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(21, 6), dpi=300)
    fig.suptitle("Features and Labels quantitative visualization", fontweight='bold')

    for k, ax in enumerate(axes.flatten()):
        ax.hist(dataset.iloc[:, k], histtype='stepfilled', alpha=0.8)
        ax.set_title(datasetCols[k])
        ax.set_ylabel("Frequency")
        title = ax.set_title("\n".join(wrap(datasetCols[k], 30)), fontdict={'fontsize': 10})
        title.set_y(1.04)
        fig.subplots_adjust(top=0.85)

        if k == 13:  #13 is the number of features
            break

    fig.tight_layout()
    plt.savefig(savingPath + "featureLabelsVisualization.jpg", dpi=300, bbox_inches='tight')



def plot_accuracies_train_val_test(acc_train, acc_val, acc_test, savingName):
    fig, ax = plt.subplots(figsize=(12, 8))

    fig.suptitle('Accuracy evolution of train, validation and test phases', fontweight='bold')
    print(acc_train)
    x = range(len(acc_train))
    plt.ylim(bottom=0.5)
    plt.plot(x, acc_train)
    plt.plot(x, acc_val)
    plt.hlines(acc_test, 0, len(acc_train), linestyle='--')

    ax.legend(['Train accuracy', 'Validation accuracy', 'Test accuracy'])

    fig.tight_layout()
    plt.savefig(savingName, dpi=300, bbox_inches='tight')
