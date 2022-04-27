from django.shortcuts import render
from .machinelearning.machinelearning import show_confusion_matrix, show_roc_curve, get_metrics
from dl.views import isDone

import xgboost as xgb
import numpy as np
import os
import joblib

# Create your views here.
def ml_index(request) :
    
    path = 'ml/data/'
    model_path = 'ml/model/'
    image_path = 'dl/static/'
    
    # load data
    test_x  = np.load(path + 'test_x.npy' )
    test_y  = np.load(path + 'test_y.npy' )
    
    # load model
    files = os.listdir(model_path)
    context = {}
    
    for file in files :
        
        if file == 'XGB.model' :
            model = xgb.XGBClassifier()
            model.load_model(model_path + file)
        else :
            model = joblib.load(model_path + file)
        model_name = file.split('.')[0]
        model_pred = model.predict(test_x)
        
        if not isDone(image_path, model_name) :
            show_confusion_matrix(model_pred, test_y, image_path, model_name)
            show_roc_curve(model_pred, test_y, image_path, model_name)
        context[model_name] = get_metrics(model_pred, test_y)
        
    return render(request, 'ml_view.html', context)