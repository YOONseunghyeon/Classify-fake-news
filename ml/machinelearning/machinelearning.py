import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score,  \
                             recall_score, f1_score, roc_curve, \
                             accuracy_score, roc_auc_score
                             
def get_metrics(y_pred, y_actual) :
    
    return {'Accuracy'  :  round(accuracy_score(y_pred, y_actual), 4),
            'Precision' : round(precision_score(y_pred, y_actual), 4),
            'Recall'    :    round(recall_score(y_pred, y_actual), 4),
            'F1-score'  :        round(f1_score(y_pred, y_actual), 4),
            'Auc'       :   round(roc_auc_score(y_pred, y_actual), 4)
            }

def show_confusion_matrix(y_pred, y_actual, path, model_name) :
    
    fig = plt.figure(figsize = (10, 8))
    plt.title(model_name + ' Confusion Matrix')
    # annot : cell에 숫자 표현 유무, fmt : 숫자 형태(d : 정수)
    # cmap : color, cbar : color bar 유무
    sns.heatmap(confusion_matrix(y_pred, y_actual), annot = True, fmt = 'd', 
                cbar = False, cmap = sns.color_palette("pastel"),
                xticklabels = ['Positive', 'Negative'],
                yticklabels = ['Positive', 'Negative'])
    plt.xlabel('Actuals')
    plt.ylabel('Predicted')
    plt.savefig(path + model_name +'_confusion_matrix.png', dpi = 300)
    
def show_roc_curve(y_pred, y_actual, path, model_name) :
    
    fig = plt.figure(figsize = (10, 8))
    FPR, TPR, n = roc_curve(y_actual, y_pred)
    
    plt.plot(FPR, TPR)
    plt.plot([0, 1], [0, 1])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.title(model_name + ' ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(path + model_name + '_ROC_curve.png', dpi = 300)