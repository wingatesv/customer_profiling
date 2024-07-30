import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from data.data_extraction import read_columns_to_extract, check_columns_presence, extract_columns

def preprocess_pt_df(df, config):
  # Read the columns to extract from the CSV file
  columns_to_extract = read_columns_to_extract(config["pt_model_columns_file"])

  if config['inference_mode']:
        columns_to_remove = ['label', 'repeat_phase_property_type']
        columns_to_extract = [col for col in columns_to_extract if col not in columns_to_remove]

  # Check for the presence of columns to extract in df
  check_columns_presence(df, columns_to_extract, "for pt modeling")
  # Extract the specified columns for df_temp_1
  df = extract_columns(df, columns_to_extract)


  if 'label' in df.columns:
    df = df[df['label'] != 0]
    df = df.drop(columns=['label'],  errors='ignore')

  if 'repeat_phase_property_type' in df.columns:
    df = df[~df['repeat_phase_property_type'].isin(['RSKU', 'Industrial'])]

    # Initialize the OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse_output=False)

    # Fit and transform the `repeat_phase_property_type` column
    one_hot_labels = one_hot_encoder.fit_transform(df[['repeat_phase_property_type']])

    # Get the names of the new one-hot encoded columns
    one_hot_label_names = one_hot_encoder.get_feature_names_out(['repeat_phase_property_type'])

    # Create a DataFrame from the one-hot encoded labels
    one_hot_labels_df = pd.DataFrame(one_hot_labels, columns=one_hot_label_names, index=df.index)

    # Drop the original `repeat_phase_property_type` column from the original DataFrame
    df = df.drop(columns=['repeat_phase_property_type'])

    # Add the one-hot encoded labels DataFrame to the original DataFrame
    df = pd.concat([df, one_hot_labels_df], axis=1)


  return df


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
    
    print("Preprocessors and models loaded")
    return model_list, preprocessor_list


def model_prediction(model, test_encoded, threshold=0.5):

  probabilities = model.predict_proba(test_encoded)[:, 1]
  test_predictions = (probabilities >= threshold).astype(int)
  return probabilities, test_predictions
 

def save_results(test_df, probabilities, test_predictions, config, mode, save_folder):
    if mode == 'landed':
      label = 'repeat_phase_property_type_Landed'
    elif mode == 'high_rise':
      label = 'repeat_phase_property_type_High Rise'
    else:
      label = 'repeat_phase_property_type_Commercial'


    if config['evaluation_mode']:
        # Prepare the output DataFrame
        prediction_df = pd.DataFrame({
            'contact_nric_masked': test_df['contact_nric_masked'],
            f"probability (threshold = 0.5)": probabilities,  # Probability of class 1
            'predicted label': test_predictions,
            'truth label': test_df[label]
        })

        output_csv_path = os.path.join(save_folder, f'{mode}_evaluation_result.csv')
        prediction_df.to_csv(output_csv_path, index=False)
        print(f"Model prediction for {mode} saved to {output_csv_path}")

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(test_df[label], test_predictions).ravel()
        accuracy = accuracy_score(test_df[label], test_predictions)
        precision = precision_score(test_df[label], test_predictions)
        recall = recall_score(test_df[label], test_predictions)
        f1 = f1_score(test_df[label], test_predictions)

        result_df = pd.DataFrame({
            'True Positive': [tp],
            'True Negative': [tn],
            'False Positive': [fp],
            'False Negative': [fn],
            'Accuracy': [round(accuracy, 4)],
            'Precision': [round(precision, 4)],
            'Recall': [round(recall, 4)],
            'F1-score': [round(f1, 4)]
        })

        result_csv_path = os.path.join(save_folder, f'{mode}_metrics_result.csv')
        result_df.to_csv(result_csv_path, index=False)
        print(f"Metrics result for {mode} saved to {result_csv_path}")

    else:
        # Prepare the output DataFrame
        prediction_df = pd.DataFrame({
            'contact_nric_masked': test_df['contact_nric_masked'],
            f"probability (threshold = 0.5)": probabilities,  # Probability of class 1
            'predicted label': test_predictions,
        })

        output_csv_path = os.path.join(save_folder, f'{mode}_inference_result.csv')
        prediction_df.to_csv(output_csv_path, index=False)
        print(f"Model prediction for {mode} saved to {output_csv_path}")

        

def property_type_test(test_df, config):
  # Create save directory if it doesn't exist
  if not os.path.exists(config['pt_result_dir']):
      os.makedirs(config['pt_result_dir'])
      print(f"Created directory: {config['pt_result_dir']}")

  # Create a new folder with today's date and time
  mode_str = 'eval' if config['evaluation_mode'] else 'infer'
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_folder = os.path.join(config['pt_result_dir'],f'{mode_str}_{timestamp}')
  os.makedirs(save_folder)
  print(f"Created directory for results: {save_folder}")

  model_list, preprocessor_list = load_model(config)

  test_df = preprocess_pt_df(test_df, config)

  test_df_clean = test_df.drop(columns=['label', 'contact_nric_masked', 'spa_date', 'repeat_phase_property_type', 'repeat_phase_property_type_Landed', 'repeat_phase_property_type_High Rise', 'repeat_phase_property_type_Commercial'], errors='ignore')

  if config['landed_mode']:
    model, preprocessor = model_list[0], preprocessor_list[0]
    test_encoded = preprocessor.transform(test_df_clean)
    probabilities, test_predictions = model_prediction(model, test_encoded)
    save_results(test_df, probabilities, test_predictions, config, 'landed', save_folder)


  
  if config['high_rise_mode']:
    model, preprocessor = model_list[1], preprocessor_list[1]
    test_encoded = preprocessor.transform(test_df_clean)
    probabilities, test_predictions = model_prediction(model, test_encoded)
    save_results(test_df, probabilities, test_predictions, config, 'high_rise', save_folder)



  if config['commercial_mode']:
    model, preprocessor = model_list[2], preprocessor_list[2]
    test_encoded = preprocessor.transform(test_df_clean)
    probabilities, test_predictions = model_prediction(model, test_encoded)
    save_results(test_df, probabilities, test_predictions, config, 'commercial', save_folder)



  if not config['landed_mode'] and config['high_rise_mode'] and config['commercial_mode']:
     warnings.warn(f"None of the property type mode is selected, please at least select one!!!")

  

  
  
