import os 
import sys

import numpy as np
import pandas as pd 
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
report = {}    
def evaluate_models(X_train, y_train,X_test,y_test,models,params):

    try:
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            best_para= gs.best_params_
            model.set_params(**best_para)
            
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            
            score= r2_score(y_pred,y_test)
            report[list(models.keys())[i]]=score

        return report
    
    except:
    
        pass