from django.http import HttpResponse
from django.shortcuts import render
from dl.deeplearning.deeplearning import DeepLearning
from tensorflow.python.keras.models import load_model

import numpy as np
import os

# Create your views here.

def isDone(image_path, model_name) :
    img_list = os.listdir(image_path)
    if model_name + '_confusion_matrix.png' in img_list and model_name + '_ROC_curve.png' in img_list :
        return True
    return False

def index(request) :
    
    path = 'C:/Users/admin/proj/dl/data/'
    model_path = 'C:/Users/admin/proj/dl/models/'
    image_path = 'C:/Users/admin/proj/dl/static/'

    train_x = np.load(path + 'train_x.npy')
    test_x  = np.load(path + 'test_x.npy' )
    train_y = np.load(path + 'train_y.npy')
    test_y  = np.load(path + 'test_y.npy' )

    models  = ['RNN', 'LSTM']
    context = {}
    
    for model in models :
        dl = DeepLearning(model, train_x, train_y, test_x, test_y)

        dl.set_model(load_model(model_path + model + '.h5'))
        dl.model_predict(test_x)
        dl.set_metrics()
        
        if not isDone(image_path, model) :
            dl.plot_roc(image_path)
            dl.plot_confusion_matrix(image_path)
        context[model] = dl.get_metrics()
  
    return render(request, 'dl_view.html', context)