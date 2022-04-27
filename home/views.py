from django.shortcuts import render
from tensorflow.python.keras.models import load_model
from .preprocessing.preprocessing import pp, pp_ml
import joblib
import xgboost as xgb
# Create your views here.

def home(request) :
    return render(request, 'home.html')

def input(request) :
    return render(request, 'input_page.html')

def get_post(request) :
    
    deep_models    = {'RNN'           : 'dl/models/RNN.h5',
                      'LSTM'          : 'dl/models/LSTM.h5',
                      'BILSTM'        : 'dl/models/BILSTM.h5',}
    machine_models = {'Random_Forest' : 'ml/model/RFC.pkl',
                      'Ada_Boost'     : 'ml/model/Ada_Boost.pkl',
                      'KNN'           : 'ml/model/KNN.pkl',
                      'LGBM'          : 'ml/model/LGBM.pkl',
                      'BNB'           : 'ml/model/BNB.pkl',
                      'SGD'           : 'ml/model/SGD.pkl',
                      'SVC'           : 'ml/model/SVC.pkl',
                      'XGB'           : 'ml/model/XGB.model',}
    
    if request.method == 'GET' :
        
        author = request.GET.get('author')
        title  = request.GET.get('title' )
        model  = request.GET.get('model' )
        
        if model in deep_models.keys() :        # deep learning 
            model_path = deep_models[model]
            model = load_model(model_path)
            pred = model.predict(pp(author + title))  # predict
            pred = round(pred[0][0] * 100, 2)
        
        elif model in machine_models.keys() :   # machine learning
            model_path = machine_models[model]
            
            if model == 'XGB' :                
                model = xgb.XGBClassifier()
                model.load_model(model_path)
            else :
                model = joblib.load(model_path)
            pred = model.predict(pp_ml(author + title))  # predict
            pred = round(pred[0] * 100, 2)
            
        context = {'prediction' : str(pred) + '%'}
        return render(request, 'result_page.html', context)

    elif request.method == 'POST' : 
        return render(request, 'result_page.html', context)
    
def hope(request) :
    return render(request, 'home_page.html')