import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# 混淆矩阵
def plotHeatMap(Y_test, Y_pred, ClassSet, norm=False, file=None,dpi=96):
    '''
    This is a function for ploting HeatMap.
    Y_test is your test labels, and Y_pred is your model's outputs.
    Especially, you need to offer classer' name which is a list, like '['a', 'b', 'c', 'd']'.
    norm means you need to normalize your results, False default.
    Y_test and Y_pred should be final output ->(None, )
    '''
    con_mat = metrics.confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(7, 7),dpi=dpi)
    # normalization, optional
    if norm == True:
        con_mat = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
        con_mat = np.around(con_mat, decimals=2)
    seaborn.heatmap(con_mat, annot=True, fmt='.2f', cmap='Blues')
    plt.xlim(0,len(ClassSet))
    plt.ylim(0,len(ClassSet))
    ticks = [i+0.5 for i in range(len(ClassSet))]
    plt.xticks(ticks,ClassSet)
    plt.yticks(ticks,ClassSet)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    # save or not, optional
    if file != None:
        plt.savefig(file,  bbox_inches='tight')
    plt.show()
    print('done')

def show_accuracy(Y_test, Y_pred, normalize=True):
    '''
    Using scikit-learn API to compute accuracy.
    Y_test and Y_pred should be final output ->(None, )
    '''
    return metrics.accuracy_score(Y_test, Y_pred, normalize=normalize)

def report(Y_test, Y_pred, labels_name=None):
    '''
    Using scikit-learn API to export report,
      include precision, recall, f1_score, support(per-class numbers), accuracy, macro vrg, weighted avg.
    Y_test and Y_pred should be final output ->(None, )
    '''
    return metrics.classification_report(Y_test, Y_pred, target_names=labels_name)

def kappa(Y_test, Y_pred):
    '''
    Using scikit-learn API to compute cohen's kappa value.
    Y_test and Y_pred should be final output ->(None, )
    '''
    return metrics.cohen_kappa_score(Y_test, Y_pred)

def f1_score(Y_test, Y_pred, average='macro'):
    '''
    Using scikit-learn API to compute f1-score value.
    Y_test and Y_pred should be final output ->(None, )
    '''
    return metrics.f1_score(Y_test, Y_pred, average=average)

def plot_ROC(Y_test, Y_prob, labels, 
            colorlist=["aqua", "darkorange", "cornflowerblue"], 
            figsize=(7, 7), dpi=96,file=None):
    '''
    Using scikit-learn API to plot ROC curve.
    Y_prob is model's outputs, not final labels.
    Y_test and Y_pred should be probability distribution output ->(None, n_classes)
    '''
    n_classes = Y_prob.shape[1]
    #Compute ROC curve and ROC area for each class.
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(Y_test[:, i], Y_prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    #Compute micro-average ROC curve and ROC area.
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(Y_test.ravel(), Y_prob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    #First aggregate all false positive rates.
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    #Then interpolate all ROC curves at this points.
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    #Finally average it and compute AUC.
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    #Plot all ROC curves.
    plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.4f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=2,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.4f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=2,
    )

    colors = cycle(colorlist)
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of {0} (area = {1:0.4f})".format(labels[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positve Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    if file != None:
        plt.savefig(file,  bbox_inches='tight')
    plt.show()
    print('done')


if __name__ == "__main__":
    '''
    output demo as follow:
    (60, 20) (60, 3)
    (15,) (15, 3)
    done
    accuracy:  0.6
                  precision    recall  f1-score   support

               a       0.56      0.71      0.63         7
               b       1.00      0.67      0.80         3
               c       0.50      0.40      0.44         5

        accuracy                           0.60        15
       macro avg       0.69      0.59      0.62        15
    weighted avg       0.63      0.60      0.60        15

    kappa:  0.3382352941176471
    f1-score:  0.6231481481481482
    f1-score:  [0.625      0.8        0.44444444]
    done
    '''
    X, y = make_classification(n_samples=60, n_classes=3, random_state=20, n_informative=3)
    y = label_binarize(y, classes=[0, 1, 2])
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    print(np.argmax(y_test,axis=1).shape,y_pred.shape)
    plotHeatMap(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1), ClassSet=['a', 'b', 'c'])
    print("accuracy: ", show_accuracy(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1)))
    print(report(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1), ['a','b','c']))
    print("kappa: ",kappa(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1)))
    print("f1-score: ", f1_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1)))
    print("f1-score: ", f1_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1), average=None))
    plot_ROC(y_test, y_prob, labels=['a','b','c'])
