import pandas as pd
import argparse
import yaml
import os
import traceback
import sys
import shutil
import warnings

from data.data_extraction import data_extraction
from data.label_generation import generate_label
from data.data_cleaning import data_cleaning
from data.feature_generation import add_features
from data.data_grouping import group_data
from data.data_bining import data_bining
from data.test_label_generation import generate_test_label
from data.validate_config import validate_config


from repeat_purchase.train import train
from repeat_purchase.test import test
from property_type.train import property_type_train
from property_type.test import property_type_test

def clear_directory(directory):
    """
    Clears all files in the specified directory.
    
    Args:
    directory (str): The directory to clear.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

                

def data_preparation(config):
  training_mode=config['training_mode']
  eval_mode=config['evaluation_mode']
  infer_mode=config['inference_mode']
  

  def read_checkpoint(file_name):
        return pd.read_csv(os.path.join(config['save_dir'], file_name), low_memory=False, dtype={config['unique_customer_id']: str})

  # Create directory if it doesn't exist
  if not os.path.exists(config['data_report_dir']):
      print("Creating data report directory.....")
      os.makedirs(config['data_report_dir'])
      print(f"Created directory: {config['data_report_dir']}")
  else:
      print(f"Directory {config['data_report_dir']} already exists. Clearing contents...")
      clear_directory(config['data_report_dir'])
      print(f"Cleared contents of directory: {config['data_report_dir']}")
  

  if config['load_df_from_checkpoint']:
        print("Loading df from checkpoint ...")
        if os.path.exists(config['save_dir']):
            files = os.listdir(config['save_dir'])

            if 'bin_df.csv' in files:
                df = read_checkpoint('bin_df.csv')
              
                print("Found bin_df.csv! continue from here....")
                return df
          

  # If no checkpoint is loaded, start from scratch
  print("Starting from scratch ... loading df from input_file")
  # Check if the input file exists
  if not os.path.exists(config['input_file']):
      raise FileNotFoundError(f"{config['input_file']} not found.")
  df = pd.read_csv(config['input_file'], low_memory=False, dtype={config['unique_customer_id']: str})

  if 'buyer_dob' not in df.columns:
        print('Feature: "buyer_dob" is not in the input_file!')
        print('Terminating program....')
        sys.exit()

  df = data_extraction(df, config)
  if training_mode or (eval_mode and not infer_mode):
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

      validate_config(config)
      
      print()
      print("------------------------------------------------------------------------------------------------------")
      print("Starting data preparation...")
      df = data_preparation(config)

      if config['training_mode'] or config['evaluation_mode']:
            if 'label' not in df.columns:
              print('Training labels are not previously generated....generating training labels')
              df = generate_label(df, config)


              output_csv_path = os.path.join(config['save_dir'], 'bin_df.csv')
              df.to_csv(output_csv_path, index=False)
              print(f"Training label generation completed! Updated: {output_csv_path}")

                

      if config['generate_test_label_mode']:
          print("Generating test labels only...")
          test_df, pt_df = generate_test_label(df, config)
          # Save the resulting DataFrame to CSV
          output_csv_path = os.path.join(config['save_dir'], 'rp_test_df.csv')
          test_df.to_csv(output_csv_path, index=False)
          print(f"Test rp label generation completed! Saved to {output_csv_path}")
          output_csv_path = os.path.join(config['save_dir'], 'pt_test_df.csv')
          pt_df.to_csv(output_csv_path, index=False)
          print(f"Test pt label generation completed! Saved to {output_csv_path}")

      else:
        # Training
        if config['training_mode']:
          if config['repeat_purchase_mode']:
            print()
            print("------------------------------------------------------------------------------------------------------")
            print("Starting repeat_purchase model training....")
            train(df, config)

          if config['property_type_mode']:
            print()
            print("------------------------------------------------------------------------------------------------------")
            print("Starting property_type model training....")
            property_type_train(df, config)

           
        if config['evaluation_mode'] and not config['inference_mode']:
          test_df, pt_df = generate_test_label(df, config)
          if config['repeat_purchase_mode']:

           
            print()
            print("------------------------------------------------------------------------------------------------------")
            print("Starting repeat_purchase model evaluation....")
            test(test_df, config)

          if config['property_type_mode']:
            print()
            print("------------------------------------------------------------------------------------------------------")
            print("Starting property_type model evaluation....")
            property_type_test(pt_df, config)
            
        elif config['inference_mode'] and not config['evaluation_mode']:
          if config['repeat_purchase_mode']:
            print()
            print("------------------------------------------------------------------------------------------------------")
            print("Starting repeat_purchase model inference....")
            test(df, config)
          if config['property_type_mode']:
            print()
            print("------------------------------------------------------------------------------------------------------")
            print("Starting property_type model inference....")
            property_type_test(df, config)
        
        
        else:
            warnings.warn(f"Cannot not perform any or perform both evaluation and inference modes at the same time!!!")
            


      print()
      print("------------------------------------------------------------------------------------------------------")
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
