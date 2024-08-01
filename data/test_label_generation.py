import pandas as pd
import os
import warnings
from data.data_extraction import read_columns_to_extract, check_columns_presence, extract_columns, convert_date_columns, filter_dataframe, remove_recent_transaction
import sys

# def generate_test_label(df, config):
#   print("Generating testing labels...")

#   # Read the columns to extract from the CSV file
#   columns_to_extract = read_columns_to_extract(config["model_training_columns_file"])
#   # Check for the presence of columns to extract in df
#   check_columns_presence(df, columns_to_extract, "for extraction in bin_df")
#   # Extract the specified columns for df_temp_1
#   df = extract_columns(df, columns_to_extract)
#   # Convert repeat_purchase from True/False to 0/1
#   df['label'] = df['label'].map({'N': 0, 'Y': 1})

#   # Exclude all Y labels
#   df = df[df['label'] != 1]

#   df['contact_nric_masked'] = df['contact_nric_masked'].astype(str)

#   if not os.path.exists(config['test_file']):
#     raise FileNotFoundError(f"{config['test_file']} not found.")

#   evaluation_year_df = pd.read_csv(config['test_file'],  dtype={'contact_nric_masked': str})

#   # Get the set of contact_nric_masked values from after_2022_df
#   evaluation_year_contact_nric = set(evaluation_year_df['contact_nric_masked'].unique())

#   # Drop the existing label column if it exists
#   if 'label' in df.columns:
#       df = df.drop(columns=['label'])

#   # Create a new label column
#   df['label'] = df['contact_nric_masked'].apply(lambda x: 1 if x in evaluation_year_contact_nric else 0)

#   # See the distribution of the label column
#   label_distribution = df['label'].value_counts()
#   print("Distribution of the test label:")
#   print(label_distribution)

#   return df


def generate_test_label(df, config):
  print("Generating testing labels...")

  if os.path.exists( os.path.join(config['save_dir'], 'for_eval_label_generation.csv')):
     print("Using split df from data extraction for label generation....")
     label_df = pd.read_csv(os.path.join(config['save_dir'], 'for_eval_label_generation.csv'), low_memory=False,  dtype={'contact_nric_masked': str})

  else:
    print(f"Cannot find for_eval_label_generation.csv for eval!")
    sys.exit()


  if label_df['sales_id'].duplicated().any():
      print("Duplicates found in sales_id..... preprocessing the raw test file")

      # Read the columns to extract from the CSV file
      columns_to_extract = read_columns_to_extract(config['data_extraction_columns_file'])
      # List of columns to convert to datetime
      date_columns = [
          'transaction_date', 'commencement_date', 'completion_date',
          'spa_stamp_date', 'sales_conversion_date', 'spa_date',
          'contact_reg_date', 'buyer_dob'
      ]

      # Check for the presence of date columns
      check_columns_presence(label_df, date_columns, "for date conversion")
      # Convert date columns to datetime
      convert_date_columns(label_df, date_columns)

      # Check for the presence of filter columns
      filter_columns = ['project_country', 'last_sales_status', 'date_type']
      check_columns_presence(label_df, filter_columns, "for filtering")

      # Filter the DataFrame based on specific conditions for df_temp_1
      filter_conditions_1 = (
          (label_df['project_country'] == 'Malaysia') &
          (label_df['last_sales_status'] == 'Sold') &
          (label_df['date_type'] == 'spa_date_type')
      )
      label_df = filter_dataframe(label_df, filter_conditions_1)

      # Remove rows with transactions within one month
      label_df = remove_recent_transaction(label_df)


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

  # Get the set of contact_nric_masked values from after_2022_df
  evaluation_year_contact_nric = set(evaluation_year_df['contact_nric_masked'].unique())

  # Drop the existing label column if it exists
  if 'label' in df.columns:
      df = df.drop(columns=['label'])

  # Create a new label column
  df['label'] = df['contact_nric_masked'].apply(lambda x: 1 if x in evaluation_year_contact_nric else 0)

  # See the distribution of the label column
  label_distribution = df['label'].value_counts()
  print("Distribution of the test label:")
  print(label_distribution)

  return df
