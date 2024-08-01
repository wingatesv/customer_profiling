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


  # Get the set of config['unique_customer_id'] values from after_2022_df
  evaluation_year_contact_nric = set(label_df[config['unique_customer_id']].unique())

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


  # Create a dictionary mapping unique_customer_id to phase_property_type from label_df
  customer_to_property_type = label_df.set_index(config['unique_customer_id'])['phase_property_type'].to_dict()

  # Create the new "repeat_phase_property_type" column in pt_df
  pt_df['repeat_phase_property_type'] = pt_df[config['unique_customer_id']].map(customer_to_property_type)
  pt_df = pt_df[~pt_df['repeat_phase_property_type'].isin(['RSKU', 'Industrial'])]
  # See the distribution of the label column
  label_distribution = pt_df['repeat_phase_property_type'].value_counts()
  print("Distribution of the repeat_phase_property_type test label:")
  print(label_distribution)


  return df, pt_df
