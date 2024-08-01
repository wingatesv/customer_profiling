import pandas as pd
import os
import warnings
import sys



def generate_test_label(df, config):
  print("Generating testing labels...")

  if os.path.exists( os.path.join(config['save_dir'], 'for_eval_label_generation.csv')):
     print("Using split df from data extraction for label generation....")
     label_df = pd.read_csv(os.path.join(config['save_dir'], 'for_eval_label_generation.csv'), low_memory=False,  dtype={config['unique_customer_id']: str})

  else:
    print(f"Cannot find for_eval_label_generation.csv for eval!")
    sys.exit()


  label_df['spa_date'] = pd.to_datetime(label_df['spa_date'], errors='coerce')

  oldest_date = label_df['spa_date'].min()
  latest_date = label_df['spa_date'].max()
  print(f"Using data for test label from {oldest_date} to {latest_date}...")

  print(f"Using data from {config['evaluation_start_date']} to {config['evaluation_end_date']} for evaluation")
  # Extract start and end dates from the config
  start_date = pd.to_datetime(config['evaluation_start_date'], errors='coerce')

  # Check for end date, if not available or out of range, use the latest date from spa_date and raise a warning
  end_date = pd.to_datetime(config['evaluation_end_date'], errors='coerce')

  # Validate dates within the range of the DataFrame
  if start_date < label_df['spa_date'].min() or start_date > label_df['spa_date'].max():
      start_date = label_df['spa_date'].min()
      warnings.warn(f"evaluation_start_date {config['evaluation_start_date']} is out of range, using the earliest date from spa_date: {start_date}")
 
  
  if end_date < label_df['spa_date'].min() or end_date > label_df['spa_date'].max():
      warnings.warn(f"evaluation_end_date {config['evaluation_end_date']} is out of range, using the latest date from spa_date: {label_df['spa_date'].max()}")
      end_date = label_df['spa_date'].max()
  
  
  # Filter the DataFrame within the date range
  mask = (label_df['spa_date'] >= start_date) & (label_df['spa_date'] <= end_date)
  evaluation_year_df = label_df.loc[mask]

  # Get the set of config['unique_customer_id'] values from after_2022_df
  evaluation_year_contact_nric = set(evaluation_year_df[config['unique_customer_id']].unique())

  # Drop the existing label column if it exists
  if 'label' in df.columns:
      df = df.drop(columns=['label'])

  # Create a new label column
  df['label'] = df[config['unique_customer_id']].apply(lambda x: 1 if x in evaluation_year_contact_nric else 0)

  # See the distribution of the label column
  label_distribution = df['label'].value_counts()
  print("Distribution of the test label:")
  print(label_distribution)

  return df
