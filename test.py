import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_model(config):
    model_dir = config['rp_model_dir']
    model_name = config['rp_model']
    bagging_split = config['bagging_split']
    
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    model_list = []

    # Check if model_dir exists
    if os.path.exists(model_dir):
        # Check if preprocessor.pkl exists
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        else:
            raise FileNotFoundError(f"{preprocessor_path} not found.")
        
        # Check and load model pickle files
        for i in range(1, bagging_split + 1):
            model_path = os.path.join(model_dir, f'{model_name}_{i}.pkl')
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                model_list.append(model)
            else:
                raise FileNotFoundError(f"{model_path} not found.")
    else:
        raise FileNotFoundError(f"{model_dir} not found.")
    
    print("Preprocessor and models loaded")
    return model_list, preprocessor


# Aggregate predictions from multiple models
def aggregate_predictions(models, test_encoded, threshold=0.5):
    all_probabilities = np.zeros((test_encoded.shape[0], len(models)))

    for i, model in enumerate(models):
        probabilities = model.predict_proba(test_encoded)
        all_probabilities[:, i] = probabilities[:, 1]

    avg_probabilities = all_probabilities.mean(axis=1)
    aggregated_predictions = (avg_probabilities >= threshold).astype(int)

    # Return both average probabilities for class 1 and the predictions
    return avg_probabilities, aggregated_predictions


def test (test_df, config):
  print("Starting repeat purchase model testing...")
  # Create save directory if it doesn't exist
  if not os.path.exists(config['result_dir']):
      os.makedirs(config['result_dir'])
      print(f"Created directory: {config['result_dir']}")

  # Create a new folder with today's date and time
  mode_str = 'eval' if config['evaluation_mode'] else 'infer'
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_folder = os.path.join(config['result_dir'],f'{mode_str}_{timestamp}')
  os.makedirs(save_folder)
  print(f"Created directory for results: {save_folder}")

  model_list, preprocessor = load_model(config)

  if config['evaluation_mode']:  

      test_df_clean = test_df.drop(columns=['label', 'contact_nric_masked'])
      test_encoded = preprocessor.transform(test_df_clean)
      # Aggregate predictions from all models
      probabilities, test_predictions = aggregate_predictions(model_list, test_encoded, config['inference_threshold'])

      # Prepare the output DataFrame
      prediction_df = pd.DataFrame({
          'contact_nric_masked': test_df['contact_nric_masked'],
          f"probability (threshold = {config['inference_threshold']})": probabilities,       # Probability of class 1
          'predicted label': test_predictions,
          'truth label': test_df['label']
      })

      output_csv_path = os.path.join(save_folder, 'evaluation_result.csv')
      prediction_df.to_csv(output_csv_path, index=False)
      print(f"Model prediction saved to {output_csv_path}")

      # Calculate metrics
      tn, fp, fn, tp = confusion_matrix(test_df['label'], test_predictions).ravel()
      accuracy = accuracy_score(test_df['label'], test_predictions)
      precision = precision_score(test_df['label'], test_predictions)
      recall = recall_score(test_df['label'], test_predictions)
      f1 = f1_score(test_df['label'], test_predictions)

      result_df = pd.DataFrame({
          'True Positive': [tp],
          'True Negative': [tn],
          'False Positive': [fp],
          'False Negative': [fn],
          'Accuracy': [round(accuracy,4)],
          'Precision': [round(precision,4)],
          'Recall': [round(recall,4)],
          'F1-score': [round(f1,4)]
      })

      result_csv_path = os.path.join(save_folder, 'metrics_result.csv')
      result_df.to_csv(result_csv_path, index=False)
      print(f"Metrics result saved to {result_csv_path}")
  
  else:
      test_df_clean = test_df.drop(columns=['label', 'contact_nric_masked'], errors = 'ignore')
      test_encoded = preprocessor.transform(test_df_clean)
      # Aggregate predictions from all models
      probabilities, test_predictions = aggregate_predictions(model_list, test_encoded, config['inference_threshold'])

      # Prepare the output DataFrame
      prediction_df = pd.DataFrame({
          'contact_nric_masked': test_df['contact_nric_masked'],
          f"probability (threshold = {config['inference_threshold']})": probabilities,       # Probability of class 1
          'predicted label': test_predictions,
      })

      output_csv_path = os.path.join(save_folder, 'inference_result.csv')
      prediction_df.to_csv(output_csv_path, index=False)
      print(f"Model prediction saved to {output_csv_path}")

  

