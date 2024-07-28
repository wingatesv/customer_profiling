import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_model(config):
    model_dir = config['pt_model_dir']
    model_name = config['pt_model']
    
    preprocessor_list = []
    model_list = []

    # Check if model_dir exists
    if os.path.exists(model_dir):
        # Check and load model pickle files
        for i in range(3):
            property_type_model_label = 'none'
            if i == 0:
              property_type_model_label = 'landed'
            elif i == 1:
              property_type_model_label = 'high_rise'
            elif i == 2:
              property_type_model_label = 'commercial'

            model_path = os.path.join(model_dir, f'{model_name}_{property_type_model_label}.pkl')
            preprocessor_path = os.path.join(model_dir, f'preprocessor_{property_type_model_label}.pkl')
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                model_list.append(model)
            else:
                raise FileNotFoundError(f"{model_path} not found.")

            if os.path.exists(preprocessor_path):
               preprocessor = joblib.load(preprocessor_path)
               preprocessor_list.append(preprocessor)
            else:
                raise FileNotFoundError(f"{preprocessor_path} not found.")
    else:
        raise FileNotFoundError(f"{model_dir} not found.")
    
    print("Preprocessor and models loaded")
    return model_list, preprocessor_list


def property_type_test(df, config):
  pass