import pandas as pd
import argparse
import yaml
import os
import traceback

from data.data_extraction import data_extraction
from data.label_generation import generate_label
from data.data_cleaning import data_cleaning
from data.feature_generation import add_features
from data.data_grouping import group_data
from data.data_bining import data_bining
from data.test_label_generation import generate_test_label

from repeat_purchase.train import train
from repeat_purchase.test import test
from property_type.train import property_type_train
# from test_pt import property_type_test


def data_preparation(config):

  def read_checkpoint(file_name):
        return pd.read_csv(os.path.join(config['save_dir'], file_name), low_memory=False, dtype={'contact_nric_masked': str})

  # Create save directory if it doesn't exist
  if not os.path.exists(config['data_report_dir']):
      print("Creating data report directory.....")
      os.makedirs(config['data_report_dir'])
      print(f"Created directory: {config['data_report_dir']}")

  if config['load_df_from_checkpoint']:
        print("Loading df from checkpoint ...")
        if os.path.exists(config['save_dir']):
            files = os.listdir(config['save_dir'])
            if 'bin_df.csv' in files:
                df = read_checkpoint('bin_df.csv')
                print("Found bin_df.csv! continue from here....")
                return df
            elif 'group_df.csv' in files:
                df = read_checkpoint('group_df.csv')
                df = data_bining(df, config)
                print("Found group_df.csv! continue from here....")
                return df
            elif 'add_feature_df.csv' in files:
                df = read_checkpoint('add_feature_df.csv')
                df = group_data(df, config)
                df = data_bining(df, config)
                print("Found add_feature_df.csv! continue from here....")
                return df
            elif  'clean_df.csv' in files:
                df = read_checkpoint('clean_df.csv')
                df = add_features(df, config)
                df = group_data(df, config)
                df = data_bining(df, config)
                print("Found clean_df.csv! continue from here....")
                return df
            elif  'label_df.csv' in files:
                df = read_checkpoint('label_df.csv')
                df = data_cleaning(df, config)
                df = add_features(df, config)
                df = group_data(df, config)
                df = data_bining(df, config)
                print("Found label_df.csv! continue from here....")
                return df
            elif 'data_extraction_df.csv' in files:
                df = read_checkpoint('data_extraction_df.csv')
                df = generate_label(df, config)
                df = data_cleaning(df, config)
                df = add_features(df, config)
                df = group_data(df, config)
                df = data_bining(df, config)
                print("Found data_extraction_df.csv! continue from here....")
                return df
  # If no checkpoint is loaded, start from scratch
  print("Starting from scratch ... loading df from input_file")
  # Check if the input file exists
  if not os.path.exists(config['input_file']):
      raise FileNotFoundError(f"{config['input_file']} not found.")
  df = pd.read_csv(config['input_file'], low_memory=False, dtype={'contact_nric_masked': str})

  df = data_extraction(df, config)
  df = generate_label(df, config)
  df = data_cleaning(df, config)
  df = add_features(df, config)
  df = group_data(df, config)
  df = data_bining(df, config)

  return df



def main(config):
    try:
      print("Running customer profiling model prediction pipeline....")

      # Create save directory if it doesn't exist
      if not os.path.exists(config['save_dir']):
          print(f"Creating saving directory....")
          os.makedirs(config['save_dir'])
          print(f"Created directory: {config['save_dir']}")

      print()
      print("------------------------------------------------------------------------------------------------------")
      print("Starting data preparation...")
      df = data_preparation(config)

      if config['train_model']:
          print()
          print("------------------------------------------------------------------------------------------------------")
          print("Starting repeat_purchase model training....")
          train(df, config)

      if config['generate_test_label_only']:
          print("Generating test labels only...")
          test_df = generate_test_label(df, config)
          # Save the resulting DataFrame to CSV
          output_csv_path = os.path.join(config['save_dir'], 'test_df.csv')
          test_df.to_csv(output_csv_path, index=False)
          print(f"Test label generation completed! Saved to {output_csv_path}")

      elif config['evaluation_mode']: 
        test_df = generate_test_label(df, config)
        print()
        print("------------------------------------------------------------------------------------------------------")
        print("Starting repeat_purchase model evaluation....")
        test(test_df, config)

      else:
        print()
        print("------------------------------------------------------------------------------------------------------")
        print("Starting repeat_purchase model inference....")
        test(df, config)

      if config['train_property_type_model']:
          print()
          print("------------------------------------------------------------------------------------------------------")
          print("Starting property_type model training....")
          property_type_train(df, config)

      # if config['test_property_type_model']:
      #     property_type_test(df=df, config=config)

      print("Customer profiling model prediction pipeline done!")



    except Exception as e:
      print(f"Error: {str(e)}")
      traceback.print_exc()
      

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Model Prediction')
    parser.add_argument('--config', type=str, help='Path to the configuration YAML file', required=True)
    
    args = parser.parse_args()
    
    # Load the configuration file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Call the main function with the parsed configurations
    main(config)