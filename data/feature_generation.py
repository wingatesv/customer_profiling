import pandas as pd
import os


def add_derived_rebate_percentage(df):

  # add rebate percentage
  df['derived_rebate_percentage'] = df['derived_non_land_posted_total_rebate'] / df['derived_non_land_posted_selling_price']

  print("Added derived_rebate_percentage")
  return df


def add_derived_township_maturity(df):

  df['spa_date'] = pd.to_datetime(df['spa_date'], errors='coerce')

  # Extract the year from transaction_date
  df['spa_year'] = df['spa_date'].dt.year

  # Group by project_name and find the earliest year for each project
  project_earliest_year = df.groupby('project_name')['spa_date'].min().dt.year.rename('earliest_project_year')

  # Merge the earliest year back to the original DataFrame
  df = df.merge(project_earliest_year, on='project_name', how='left')

  # Define the function to determine township maturity
  def determine_maturity(row):
      year_diff = row['spa_year'] - row['earliest_project_year']
      if year_diff < 5:
          return "NEW"
      elif 5 <= year_diff <= 10:
          return "DEVELOPING"
      else:
          return "MATURED"

  # Apply the function to create the derived_township_maturity column
  df['derived_township_maturity'] = df.apply(determine_maturity, axis=1)

  # Drop intermediate columns if necessary
  df.drop(columns=['spa_year', 'earliest_project_year'], inplace=True)

  print("Added derived_township_maturity")
  return df


def add_derived_completion_status(df):

  df['spa_year'] = df['spa_date'].dt.year
  # Group by phase and find the earliest year for each phase
  phase_earliest_year = df.groupby('phase')['spa_date'].min().dt.year.rename('earliest_phase_year').reset_index()

  # Merge the earliest year back to the original DataFrame
  df = df.merge(phase_earliest_year, on='phase', how='left')

  # Define the function to determine completion status
  def determine_completion_status(row):
      year_diff = row['spa_year'] - row['earliest_phase_year']
      if row['derived_phase_property_type'] in ['LANDED', 'RSKU']:
          return "IN PROGRESS" if year_diff < 3 else "COMPLETED"
      elif row['derived_phase_property_type'] == 'HIGH RISE':
          return "IN PROGRESS" if year_diff < 4 else "COMPLETED"
      elif row['derived_phase_property_type'] in ['COMMERCIAL', 'INDUSTRIAL']:
          return "IN PROGRESS" if year_diff < 2 else "COMPLETED"
      else:
          return "UNKNOWN"  # In case there are other types not specified

  # Apply the function to create the derived_completion_status column
  df['derived_completion_status'] = df.apply(determine_completion_status, axis=1)

  # Optionally, drop intermediate columns if necessary
  df.drop(columns=['spa_year', 'earliest_phase_year'], inplace=True)

  print("Added derived_completion_status")
  return df


def add_features(df, config):
  print("Adding features in progress...")
  save_dir=config['save_dir']
  save_csv=config['save_output']

  df = add_derived_rebate_percentage(df)
  df = add_derived_township_maturity(df)
  df = add_derived_completion_status(df)

  print("Feature generation phase completed!")
  if save_csv:
      if save_dir is None:
          raise ValueError("save_dir must be provided if save_csv is True.")
      
      # Ensure the save directory exists
      os.makedirs(save_dir, exist_ok=True)

      # Save the resulting DataFrame to CSV
      output_csv_path = os.path.join(save_dir, 'add_feature_df.csv')
      df.to_csv(output_csv_path, index=False)
      print(f"Feature generation and saving completed! Saved to {output_csv_path}")

  return df
