import os
import pandas as pd
import numpy as np
import joblib
import shutil
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier

from data.data_extraction import read_columns_to_extract, check_columns_presence, extract_columns
from repeat_purchase.train import encode_features, train_model, get_feature_importances


def generate_one_hot_label(df):
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

  # Extract the columns for commercial_df
  commercial_df = df.drop(columns=[
      'repeat_phase_property_type_High Rise', 'repeat_phase_property_type_Landed' , 'derived_phase_property_type'
  ]).rename(columns={'repeat_phase_property_type_Commercial': 'label'})

  # Extract the columns for high_rise_df
  high_rise_df = df.drop(columns=[
      'repeat_phase_property_type_Commercial', 'repeat_phase_property_type_Landed', 'derived_phase_property_type'
  ]).rename(columns={'repeat_phase_property_type_High Rise': 'label'})

  # Extract the columns for landed_df
  landed_df = df.drop(columns=[
      'repeat_phase_property_type_Commercial', 'repeat_phase_property_type_High Rise', 'derived_phase_property_type'
  ]).rename(columns={'repeat_phase_property_type_Landed': 'label'})


  df_list = [landed_df, high_rise_df, commercial_df]
  return df_list


def preprocess_pt_df(df, config):
  # Read the columns to extract from the CSV file
  columns_to_extract = read_columns_to_extract(config["pt_model_columns_file"])
  columns_to_extract = columns_to_extract.append(config['unique_customer_id'])
  # Check for the presence of columns to extract in df
  check_columns_presence(df, columns_to_extract, "for pt modeling")
  # Extract the specified columns for df_temp_1
  df = extract_columns(df, columns_to_extract)


  if 'label' in df.columns:
    df = df[df['label'] != 0]
    df = df.drop(columns=['label'],  errors='ignore')

  if 'repeat_phase_property_type' in df.columns:
    df = df[~df['repeat_phase_property_type'].isin(['RSKU', 'Industrial'])]

  return df


def property_type_train(df, config):
  if os.path.exists(config['pt_model_dir']):
    # Remove all files in the directory
    print("Found existing directory...delete all files for model training")
    for filename in os.listdir(config['pt_model_dir']):
        file_path = os.path.join(config['pt_model_dir'], filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
  else:
      os.makedirs(config['pt_model_dir'])
      print(f"Created directory: {config['pt_model_dir']}")

  df = preprocess_pt_df(df, config)
  
  # See the distribution of the label column
  label_distribution = df['repeat_phase_property_type'].value_counts()
  print("Distribution of the property_type label:")
  print(label_distribution)

  df_list = generate_one_hot_label(df)

  property_type_model_label = 'none'

  for i, property_df in enumerate(df_list):
      if i == 0:
        property_type_model_label = 'landed'
      elif i == 1:
         property_type_model_label = 'high_rise'
      elif i == 2:
        property_type_model_label = 'commercial'

      # Perform train-validation split with 20% of the train set used as validation set and fixed seed
      train_df, val_df = train_test_split(property_df, test_size=config['pt_validation_split'], random_state=42)

      # Separate the target variable
      train_target = train_df['label']
      val_target = val_df['label']

      # Drop the target variable from the feature set
      train_df_2 = train_df.drop(columns=['label', config['unique_customer_id'],  'spa_date'], errors='ignore')
      val_df_2 = val_df.drop(columns=['label', config['unique_customer_id'],  'spa_date'], errors='ignore')

      # Encode the training features
      train_encoded, preprocessor, categorical_features, numerical_features = encode_features(train_df_2)

      # Use the fitted preprocessor to transform the test data
      val_encoded = preprocessor.transform(val_df_2)

      # Automatically compute class weights
      class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=train_target)
      class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

      # Train and evaluate the model
      model, report = train_model(config['pt_model'], train_encoded, train_target, val_encoded, val_target, class_weight=class_weight_dict)

      # Get feature importances if possible        
      feature_importances, feature_names = get_feature_importances(model, train_encoded, preprocessor, categorical_features, numerical_features, config['pt_model'])

      print(f"Model: {config['pt_model']}_{property_type_model_label}")
      print("Validation Classification Report:")
      print(report)

      output_preprocessor_path = os.path.join(config['pt_model_dir'], f'preprocessor_{property_type_model_label}.pkl')
      joblib.dump(preprocessor, output_preprocessor_path)
      print(f"Save preprocessor in {output_preprocessor_path}...")

      output_model_path = os.path.join(config['pt_model_dir'], f"{config['pt_model']}_{property_type_model_label}.pkl")
      joblib.dump(model, output_model_path)
      print(f"Save model in {output_model_path}...")


      # Create a DataFrame for better visualization
      feature_importances_df = pd.DataFrame({
          'Feature': feature_names,
          'Importance': feature_importances
      })
      feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
      output_csv_path = os.path.join(config['pt_model_dir'], f'feature_importance_{property_type_model_label}.csv')
      feature_importances_df.to_csv(output_csv_path, index=False)
      print(f"Model prediction saved to {output_csv_path}")






  
