import pandas as pd
import os
import warnings
import sys



def generate_test_label(df, config):
  print("Generating testing labels...")

  df = df[df['label'] != 1]
  df[config['unique_customer_id']] = df[config['unique_customer_id']].astype(str)

  if os.path.exists( os.path.join(config['save_dir'], 'for_eval_label_generation.csv')):
     print("Using split df from data extraction for label generation....")
     label_df = pd.read_csv(os.path.join(config['save_dir'], 'for_eval_label_generation.csv'), low_memory=False,  dtype={config['unique_customer_id']: str})

  else:
    print(f"Cannot find for_eval_label_generation.csv for eval!")
    sys.exit()


  label_df['spa_date'] = pd.to_datetime(label_df['spa_date'], errors='coerce')

  
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

  pt_df = df[df['label'] != 0]
  pt_df = pt_df.drop(columns=['label'],  errors='ignore')
  pt_df.to_csv('/content/pt_df', index=False)

  return df, pt_df
