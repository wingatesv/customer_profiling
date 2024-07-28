import os
import pandas as pd

def group_salutation(df, feature_mapping_dir):

  file_path = os.path.join(feature_mapping_dir, 'feature_mapping - salutation.csv')
  # Check if the file path exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")

  # Step 1: Read the feature_mapping_salutation.csv into a DataFrame
  feature_mapping_salutation_df = pd.read_csv(file_path)

  # Step 2: Clean and prepare the salutation values
  df['clean_salutation'] = df['derived_contact_salutation'].str.strip().str.upper()


  #  Ensure feature_mapping_salutation_df also has clean and uppercase values
  feature_mapping_salutation_df['clean_salutation'] = feature_mapping_salutation_df['clean_salutation'].str.strip().str.upper()


  # Step 3: Merge extracted_df with feature_mapping_salutation_df on the cleaned salutation
  merged_df = pd.merge(df, feature_mapping_salutation_df, on='clean_salutation',  how='left')


  # Step 4: Check for unmapped values
  unmapped_values = merged_df[merged_df['map_salutation'].isnull()]['clean_salutation'].unique()
  if len(unmapped_values) > 0:
      raise ValueError(f"Error: The following salutation values do not have mapping targets: {unmapped_values}")

  # Step 5: Replace derived_contact_salutation values with map_salutation values
  df['group_derived_contact_salutation'] = merged_df['map_salutation']

  # Drop the temporary 'clean_salutation' column
  df = df.drop(columns=['clean_salutation'])

  print("Done grouping salutation")
  return df

def group_nationality(df, feature_mapping_dir):

  file_path = os.path.join(feature_mapping_dir, 'feature_mapping - nationality.csv')
  # Check if the file path exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")

  # Step 2: Read the feature_mapping-nationality.csv into a DataFrame
  feature_mapping_nationality_df = pd.read_csv(file_path)

  # Clean and prepare the nationality values
  df['clean_nationality'] = df['derived_nationality'].str.strip().str.upper()

  # Merge extracted_df with feature_mapping_nationality_df on the cleaned nationality
  merged_df_nationality = pd.merge(df, feature_mapping_nationality_df, left_on='clean_nationality', right_on='clean_nationality', how='left')

  # Check for unmapped nationality values
  unmapped_nationalities = merged_df_nationality[merged_df_nationality['map_nationality'].isnull()]['clean_nationality'].unique()
  if len(unmapped_nationalities) > 0:
      raise ValueError(f"Error: The following nationality values do not have mapping targets: {unmapped_nationalities}")

  # Replace derived_nationality values with map_nationality values
  df['group_derived_nationality'] = merged_df_nationality['map_nationality']

  # Drop the temporary 'clean_nationality' column
  df = df.drop(columns=['clean_nationality'])


  print("Done grouping nationality")
  return df

def group_occupation(df, feature_mapping_dir):

  file_path = os.path.join(feature_mapping_dir, 'feature_mapping - occupation.csv')
  # Check if the file path exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")

  # Step 3: Read the feature_mapping-occupation.csv into a DataFrame
  feature_mapping_occupation_df = pd.read_csv(file_path)

  # Clean and prepare the occupation values
  df['clean_occupation'] = df['derived_contact_occupation'].str.strip().str.upper()

  # Merge extracted_df with feature_mapping_occupation_df on the cleaned occupation
  merged_df_occupation = pd.merge(df, feature_mapping_occupation_df, left_on='clean_occupation', right_on='clean_occupation', how='left')

  # Check for unmapped occupation values
  unmapped_occupations = merged_df_occupation[merged_df_occupation['map_occupation'].isnull()]['clean_occupation'].unique()
  if len(unmapped_occupations) > 0:
      raise ValueError(f"Error: The following occupation values do not have mapping targets: {unmapped_occupations}")

  # Replace derived_contact_occupation values with map_occupation values
  df['derived_occupation_status'] = merged_df_occupation['map_occupation']

  # Drop the temporary 'clean_occupation' column
  df = df.drop(columns=['clean_occupation'])


  print("Done grouping occupation")
  return df

def group_financier(df, feature_mapping_dir):

  file_path = os.path.join(feature_mapping_dir, 'feature_mapping - financier.csv')
  # Check if the file path exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")

  # Step 4: Read the feature_mapping-financier.csv into a DataFrame
  feature_mapping_financier_df = pd.read_csv(file_path)

  # Clean and prepare the financier name values
  df['clean_financier1_name'] = df['derived_financier_name'].str.strip().str.upper()

  # Merge extracted_df with feature_mapping_financier_df on the cleaned financier name
  merged_df_financier = pd.merge(df, feature_mapping_financier_df, left_on='clean_financier1_name', right_on='clean_financier1_name', how='left')

  # Check for unmapped financier name values
  unmapped_financiers = merged_df_financier[merged_df_financier['map_financier1_name'].isnull()]['clean_financier1_name'].unique()
  if len(unmapped_financiers) > 0:
      raise ValueError(f"Error: The following financier name values do not have mapping targets: {unmapped_financiers}")

  # Replace derived_financier_name values with map_financier1_name values
  df['group_derived_financier_name'] = merged_df_financier['map_financier1_name']

  # Drop the temporary 'clean_financier1_name' column
  df = df.drop(columns=['clean_financier1_name'])

  print("Done grouping financier_name")
  return df

def group_race(df, feature_mapping_dir):

  file_path = os.path.join(feature_mapping_dir, 'feature_mapping - race.csv')
  # Check if the file path exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")

 # Step 5: Read the feature_mapping-race.csv into a DataFrame
  feature_mapping_race_df = pd.read_csv(file_path)

  # Clean and prepare the race values
  df['clean_race'] = df['derived_race'].str.strip().str.upper()

  # Merge extracted_df with feature_mapping_race_df on the cleaned race
  merged_df_race = pd.merge(df, feature_mapping_race_df, left_on='clean_race', right_on='clean_race', how='left')

  # Check for unmapped race values
  unmapped_races = merged_df_race[merged_df_race['map_race'].isnull()]['clean_race'].unique()
  if len(unmapped_races) > 0:
      raise ValueError(f"Error: The following race values do not have mapping targets: {unmapped_races}")

  # Replace derived_race values with map_race values
  df['group_derived_race'] = merged_df_race['map_race']

  # Drop the temporary 'clean_race' column
  df = df.drop(columns=['clean_race'])

  print("Done grouping race")
  return df

def group_contact_city(df, feature_mapping_dir):

  file_path = os.path.join(feature_mapping_dir, 'feature_mapping - city.csv')
  # Check if the file path exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")

  # Step 1: Read the feature_mapping-city.csv into a DataFrame
  feature_mapping_city_df = pd.read_csv(file_path)

  # Trim and uppercase the derived_contact_city values
  df['clean_city'] = df['derived_contact_city']

  # Step 3: Merge extracted_df with feature_mapping_city_df on the cleaned city name
  merged_df = pd.merge(df, feature_mapping_city_df, on='clean_city', how='left')

  # Step 4: Check for unmapped values
  unmapped_values = merged_df[merged_df['map_city_tier1'].isnull()]['clean_city'].unique()
  if len(unmapped_values) > 0:
      raise ValueError(f"Error: The following salutation values do not have mapping targets: {unmapped_values}")

  # Step 4: Replace derived_contact_city values with map_city_tier_1 values
  df['group_derived_contact_city'] = merged_df['map_city_tier1']

  # Drop the temporary 'clean_city' column
  df = df.drop(columns=['clean_city'])

  print("Done grouping contact_city")
  return df

def group_annual_income(df, feature_mapping_dir):

  file_path = os.path.join(feature_mapping_dir, 'feature_mapping - annual_income.csv')
  # Check if the file path exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")

  # Read the feature_mapping-annual_income.csv into a DataFrame
  feature_mapping_annual_income_df = pd.read_csv(file_path)

  # Clean and prepare the annual income values
  df['clean_contact_annual_income'] = df['derived_contact_annual_income'].str.strip().str.upper()

  # Merge extracted_df with feature_mapping_annual_income_df on the cleaned annual income
  merged_df_annual_income = pd.merge(df, feature_mapping_annual_income_df, on='clean_contact_annual_income', how='left')

  # Check for unmapped annual income values
  unmapped_annual_incomes = merged_df_annual_income[merged_df_annual_income['map_annual_income'].isnull()]['clean_contact_annual_income'].unique()
  if len(unmapped_annual_incomes) > 0:
      raise ValueError(f"Error: The following annual income values do not have mapping targets: {unmapped_annual_incomes}")

  # Apply the mapping or set to 'Not Applicable' based on the condition
  df['group_derived_contact_annual_income'] = df.apply(
      lambda row: 'NOT APPLICABLE' if row['derived_contact_identification_type'] == 'REG NO.' else merged_df_annual_income.loc[row.name, 'map_annual_income'],
      axis=1
  )

  # Drop the temporary 'clean_annual_income' column
  df = df.drop(columns=['clean_contact_annual_income'])

  print("Done grouping annual_income")
  return df


def group_data(df, config):
  print("Data grouping in progress...")
  feature_mapping_dir=config['feature_mapping_dir']
  save_dir=config['save_dir']
  save_csv=config['save_output']

  if feature_mapping_dir == None:
      raise ValueError(f"Derived data directory is missing")

  df = group_salutation(df, feature_mapping_dir)
  df = group_nationality(df, feature_mapping_dir)
  df = group_occupation(df, feature_mapping_dir)
  df = group_financier(df, feature_mapping_dir)
  df = group_race(df, feature_mapping_dir)
  df = group_contact_city(df, feature_mapping_dir)
  df = group_annual_income(df, feature_mapping_dir)


  print("Data grouping phase completed!")
  if save_csv:
      if save_dir is None:
          raise ValueError("save_dir must be provided if save_csv is True.")
      
      # Ensure the save directory exists
      os.makedirs(save_dir, exist_ok=True)

      # Save the resulting DataFrame to CSV
      output_csv_path = os.path.join(save_dir, 'group_df.csv')
      df.to_csv(output_csv_path, index=False)
      print(f"Data grouping and saving completed! Saved to {output_csv_path}")

  return df
