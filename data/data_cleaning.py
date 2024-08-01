import pandas as pd
import numpy as np
import os

def write_report(report_content, report_file_path):
    """
    Writes the given report content to the specified file path.
    
    Args:
    report_content (str): The content to write to the report.
    report_file_path (str): The path to the report file.
    """
    
    # Create the file if it does not exist, then append the content
    if not os.path.exists(report_file_path):
        with open(report_file_path, 'w') as report_file:
            report_file.write(report_content)
    else:
        with open(report_file_path, 'a') as report_file:
            report_file.write(report_content)

def clean_id_type(df, derived_data_dir, report_file_path):
  
    file_path = os.path.join(derived_data_dir, 'derived_id_type.csv')
    # Check if the file path exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read the CSV file into a DataFrame
    id_df = pd.read_csv(file_path, low_memory=False)

    # Step 1: Create the new column 'derived_contact_identification_type' and set it to the values of 'contact_identification_type'
    df['derived_contact_identification_type'] = df['contact_identification_type']

    # Step 2: Update 'derived_contact_identification_type' in df using 'derived_id_type' from id_df based on 'sales_id'
    df = df.merge(id_df[['sales_id', 'derived_id_type']], on='sales_id', how='left', suffixes=('', '_update'))

    # Step 3: Impute "Undefined" values in 'derived_contact_identification_type' with 'derived_id_type'
    df['derived_contact_identification_type'] = df.apply(
        lambda row: row['derived_id_type'] if row['derived_contact_identification_type'] == 'Undefined' else row['derived_contact_identification_type'],
        axis=1
    )

    # Drop the temporary 'derived_id_type' column
    df.drop(columns=['derived_id_type'], inplace=True)

    # Strip and convert to upper case
    df['derived_contact_identification_type'] = df['derived_contact_identification_type'].str.strip().str.upper()

    # Condition for derived_contact_identification_type being 'OTHERS' or null
    condition_others_or_null = df['derived_contact_identification_type'].isnull() | (df['derived_contact_identification_type'] == 'OTHERS')

    # Impute based on the contact_purchase_type and customer_nationality
    def impute_identification_type(row):
        if row['contact_purchaser_type'] == 'Corporate':
            return 'REG NO.'
        elif row['contact_purchaser_type'] in ['Individual', 'Undefined']:
            if str(row['customer_nationality']).strip().upper() == 'MALAYSIA':
                return 'NRIC NO.'
            else:
                return 'PASSPORT'
        return row['derived_contact_identification_type']

    # Apply the imputation logic to the rows that meet the condition
    df.loc[condition_others_or_null, 'derived_contact_identification_type'] = df[condition_others_or_null].apply(impute_identification_type, axis=1)

    # Additional step to replace 'REG NO' with 'REG NO.'
    df['derived_contact_identification_type'] = df['derived_contact_identification_type'].replace('REG NO', 'REG NO.')

    if df['derived_contact_identification_type'].isnull().any():
      null_row_number = df['derived_contact_identification_type'].isnull().sum()
      report_content = f'derived_contact_identification_type - Still contain {null_row_number} null rows after cleaning, all null rows removed'
      write_report(report_content, report_file_path)
      # Remove rows where derived_contact_identification_type is null
      df = df[df['derived_contact_identification_type'].notnull()]

    print('Cleaned contact_identifcation_type')

    return df

def clean_built_up_area_sqft(df, derived_data_dir, report_file_path):

  file_path = os.path.join(derived_data_dir, 'built_up_area_range.csv')
  # Check if the file path exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")

  # Load the built_up_area_range.csv into a DataFrame
  built_up_area_range_df = pd.read_csv(file_path)

  # Clean the build_type columns in both DataFrames
  df['clean_build_type'] = df['build_type'].str.strip().str.upper()
  built_up_area_range_df['clean_build_type'] = built_up_area_range_df['build_type'].str.strip().str.upper()

  # Merge the dataframes based on project_name, phase, and clean_build_type
  merged_df = pd.merge(df, built_up_area_range_df,
                      left_on=['project_name', 'phase', 'clean_build_type'],
                      right_on=['project_name', 'phase', 'clean_build_type'],
                      suffixes=('', '_range'),
                      how='left', indicator=True)

  # Create a new column in extracted_df
  df['derived_built_up_area_sqft'] = np.nan

  # Define the tolerance
  tolerance = 0.2

  # Apply logic only to matched rows
  matched_df = merged_df[merged_df['_merge'] == 'both']
  unmatched_df = merged_df[merged_df['_merge'] != 'both']

  # Iterate over the rows of the matched DataFrame to populate derived_built_up_area_sqft
  for index, row in matched_df.iterrows():
      if pd.notnull(row['built_up_area_sqft']) and row['built_up_area_sqft'] > 0:
          if row['min_area_transformed_to_sqft'] * (1 - tolerance) <= row['built_up_area_sqft'] <= row['max_area_transformed_to_sqft'] * (1 + tolerance):
              # print('Got built_up_area_sqft and within range')
              derived_value = row['built_up_area_sqft']
              # print(f"{row['clean_build_type']}, Derived built_up_area_sqft: {derived_value}, min: {row['min_area_transformed_to_sqft']}, max: {row['max_area_transformed_to_sqft']}")
          else:
              derived_value = row['built_up_area_sqft'] * 0.092903
              # print('Got built_up_area_sqft but not within range')
              # print(f"{row['clean_build_type']}, Original built_up_area_sqft: {row['built_up_area_sqft']}, Derived built_up_area_sqft: {derived_value}, min: {row['min_area_transformed_to_sqft']}, max: {row['max_area_transformed_to_sqft']}")
      elif pd.notnull(row['built_up_area_sqm']) and row['built_up_area_sqm'] > 0:
          if row['min_area_transformed_to_sqm'] * (1 - tolerance) <= row['built_up_area_sqm'] <= row['max_area_transformed_to_sqm'] * (1 + tolerance):
              derived_value = row['built_up_area_sqm'] * 10.7639
              # print('Got built_up_area_sqm and within range')
              # print(f"{row['clean_build_type']}, Built_up_area_sqm: {row['built_up_area_sqm']}, Derived built_up_area_sqft: {derived_value}, min: {row['min_area_transformed_to_sqm']}, max: {row['max_area_transformed_to_sqm']}")
          else:
              derived_value = row['built_up_area_sqm']
              # print('Got built_up_area_sqm but not within range')
              # print(f"{row['clean_build_type']}, Built_up_area_sqm: {row['built_up_area_sqm']}, Derived built_up_area_sqft: {derived_value}, min: {row['min_area_transformed_to_sqm']}, max: {row['max_area_transformed_to_sqm']}")
      else:
          if pd.notnull(row['mean_sqft']) and row['mean_sqft'] > 0:
              # print(f"{row['clean_build_type']}, No built_up_area_sqft and built_up_area_sqm, so use mean_sqft")
              derived_value = row['mean_sqft']
          else:
              # print(f"{row['sales_id']}, No built_up_area_sqft and built_up_area_sqm, and no matched mean_sqft")
              derived_value = np.nan

      df.at[row.name, 'derived_built_up_area_sqft'] = derived_value

  # Calculate mean derived_built_up_area_sqft for each clean_build_type in matched rows
  mean_derived_built_up_area_sqft = df.groupby('clean_build_type')['derived_built_up_area_sqft'].mean().to_dict()
  # Print the distribution of the mean derived_built_up_area_sqft based on build_type
  # print(mean_derived_built_up_area_sqft)
  # Set derived_built_up_area_sqft for unmatched rows based on the mean values
  for index, row in unmatched_df.iterrows():
      if row['clean_build_type'] in mean_derived_built_up_area_sqft and pd.notnull(mean_derived_built_up_area_sqft[row['clean_build_type']]):
          derived_value = mean_derived_built_up_area_sqft[row['clean_build_type']]
      else:
          if pd.notnull(row['built_up_area_sqft']) and row['built_up_area_sqft'] > 0 and row['built_up_area_sqft'] >= 250:
              derived_value = row['built_up_area_sqft']
          elif pd.notnull(row['built_up_area_sqm']) and row['built_up_area_sqm'] > 0:
              if row['built_up_area_sqm'] <= 1000:
                  derived_value = row['built_up_area_sqm'] * 10.7639
              else:
                  derived_value = row['built_up_area_sqm']
              # print(f"Sales ID: {row['sales_id']}, Built_type: {row['clean_build_type']}, Built_up_area_sqm: {row['built_up_area_sqm']}, Derived built_up_area_sqft: {derived_value}")
          else:
              if pd.notnull(row['land_area_sqft']) and row['land_area_sqft'] > 0:
                  if row['land_area_sqft'] >= 1000:
                      derived_value = row['land_area_sqft']
                  else:
                      derived_value = row['land_area_sqft'] * 10.7639
              elif  pd.notnull(row['land_area_sqm']) and row['land_area_sqm'] > 0:
                  if row['land_area_sqm'] >= 1500:
                      derived_value = row['land_area_sqm']
                  else:
                      derived_value = row['land_area_sqm'] * 0.092903
              else:
                  derived_value = np.nan
                  # print(f"Sales ID: {row['sales_id']}, Build_type: {row['clean_build_type']}, land area sqft: {row['land_area_sqft']}, land area sqm: {row['land_area_sqm']}")

      df.at[row.name, 'derived_built_up_area_sqft'] = derived_value

  # Impute or print information for rows with 0, NaN, or missing derived_built_up_area_sqft
  for index, row in df.iterrows():
      if pd.isnull(row['derived_built_up_area_sqft']) or row['derived_built_up_area_sqft'] == 0:
          build_type = row['clean_build_type']
          if build_type in mean_derived_built_up_area_sqft and not pd.isnull(mean_derived_built_up_area_sqft[build_type]):
              df.at[index, 'derived_built_up_area_sqft'] = mean_derived_built_up_area_sqft[build_type]
          
              # print(f"Sales ID: {row['sales_id']}, Build_type: {row['clean_build_type']}, land area sqft: {row['land_area_sqft']}, land area sqm: {row['land_area_sqm']}")

  # Update rows in extracted_df where derived_built_up_area_sqft is greater than 0 but less than 200
  below_200_condition = (df['derived_built_up_area_sqft'] > 0) & (df['derived_built_up_area_sqft'] < 200)

  # Apply the update with a condition to check for NaN in the mean value
  for index, row in df[below_200_condition].iterrows():
      build_type = row['clean_build_type']
      mean_value = mean_derived_built_up_area_sqft.get(build_type, np.nan)
      if not pd.isnull(mean_value):
          df.at[index, 'derived_built_up_area_sqft'] = mean_value
      else:
          df.at[index, 'derived_built_up_area_sqft'] = row['derived_built_up_area_sqft'] * 10.7639

  if df['derived_built_up_area_sqft'].isnull().any():
      null_row_number = df['derived_built_up_area_sqft'].isnull().sum()
      report_content = f'derived_built_up_area_sqft - Still contain {null_row_number} null rows after cleaning, all null rows removed'
      write_report(report_content, report_file_path)
      df = df[df['derived_built_up_area_sqft'].notnull()]

  

  print("Cleaned built_up_area_sqft")
  return df

def clean_bumi_status(df, report_file_path):

  df['derived_bumi_status'] = df['customer_bumi_status'].str.upper()
  print('Cleaned customer_bumi_status')

  if df['derived_bumi_status'].isnull().any():
    null_row_number = df['derived_bumi_status'].isnull().sum()
    report_content = f'derived_bumi_status - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_bumi_status'].notnull()]

  return df

def clean_build_type(df, report_file_path):

  df['derived_build_type'] = df['build_type'].str.upper()
  

  if df['derived_build_type'].isnull().any():
    null_row_number = df['derived_build_type'].isnull().sum()
    report_content = f'derived_build_type - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_build_type'].notnull()]

  print('Cleaned build_type')
  return df

def clean_phase_property_type(df, report_file_path):

  df['derived_phase_property_type'] = df['phase_property_type'].str.upper()

  if df['derived_phase_property_type'].isnull().any():
    null_row_number = df['derived_phase_property_type'].isnull().sum()
    report_content = f'derived_phase_property_type - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_phase_property_type'].notnull()]

  print('Cleaned phase_property_type')

  return df

def clean_gender(df, report_file_path):
    def impute_gender(row):
       
        id_type = row['derived_contact_identification_type']
        if id_type == 'REG NO.':
            return 'NOT APPLICABLE'
        elif id_type in ['NRIC NO.', 'PASSPORT']:
            if row['customer_gender'].strip().upper() not in ['UNDEFINED', 'NOT APPLICABLE', 'OTHERS', '-']:
                return str(row['customer_gender']).upper()
            elif row['contact_gender'].strip().upper() not in ['UNDEFINED', 'NOT APPLICABLE', 'OTHERS', '-']:
                return str(row['contact_gender']).upper()
            elif str(row['contact_nric'])[-1].isdigit():
                return 'MALE' if int(str(row['contact_nric'])[-1]) % 2 != 0 else 'FEMALE'
            else:
                return df['customer_gender'].mode()[0].upper()  # Impute with the mode of customer_gender
        else:
          return df['customer_gender'].mode()[0].upper()

    # Apply the gender imputation logic
    df['derived_gender'] = df.apply(impute_gender, axis=1)

    if df['derived_gender'].isnull().any():
      null_row_number = df['derived_gender'].isnull().sum()
      report_content = f'derived_gender - Still contain {null_row_number} null rows after cleaning, all null rows removed'
      write_report(report_content, report_file_path)
      df = df[df['derived_gender'].notnull()]

    print('Cleaned gender')
    return df


def clean_marital_status(df, report_file_path):
    def impute_marital(row):
      if row['derived_contact_identification_type'] == 'REG NO.':
          return 'NOT APPLICABLE'
      elif row['derived_contact_identification_type'] in ['NRIC NO.', 'PASSPORT']:
          if row['customer_marital_status'].strip().upper() not in ['UNDEFINED', 'NOT APPLICABLE', 'OTHER', '-']:
              return str(row['customer_marital_status']).upper()
          elif row['contact_marital_status'].strip().upper() not in ['UNDEFINED', 'NOT APPLICABLE', 'OTHER', '-']:
              return str(row['contact_marital_status']).upper()
          else:
              return df['customer_marital_status'].mode()[0].upper()
      else:
        return df['customer_marital_status'].mode()[0].upper()


    df['derived_marital_status'] = df.apply(impute_marital, axis=1)

    if df['derived_marital_status'].isnull().any():
      null_row_number = df['derived_marital_status'].isnull().sum()
      report_content = f'derived_marital_status - Still contain {null_row_number} null rows after cleaning, all null rows removed'
      write_report(report_content, report_file_path)
      df = df[df['derived_marital_status'].notnull()]


    print('Cleaned marital_status')
    return df

def clean_salutation(df, report_file_path):
    def determine_salutation(row):

        if row['derived_contact_identification_type'] == 'REG NO.':
              return 'Not Applicable'.upper()
        if row['contact_salutation'] in ['Not Applicable', 'Undefined', 'Others', '-'] or pd.isnull(row['contact_salutation']):
            if row['derived_contact_identification_type'] in ['NRIC NO.', 'PASSPORT']:
                if row['derived_gender'] == 'MALE':
                    return 'Mr'.upper()
                elif row['derived_gender'] == 'FEMALE':
                    return 'Ms'.upper()
        return str(row['contact_salutation']).upper()

    # Apply the function to create derived_contact_salutation
    df['derived_contact_salutation'] = df.apply(determine_salutation, axis=1)

    if df['derived_contact_salutation'].isnull().any():
      null_row_number = df['derived_contact_salutation'].isnull().sum()
      report_content = f'derived_contact_salutation - Still contain {null_row_number} null rows after cleaning, all null rows removed'
      write_report(report_content, report_file_path)
      df = df[df['derived_contact_salutation'].notnull()]

    print('Cleaned salutation')
    return df

def clean_purchaser_type(df, report_file_path):
    def impute_purchaser_type(row):
    # if row['contact_purchaser_type'] == 'Undefined':
        if row['derived_contact_identification_type'] in ['NRIC NO.', 'PASSPORT']:
            return 'Individual'.upper()
        elif row['derived_contact_identification_type'] == 'REG NO.':
            return 'Corporate'.upper()

    # Apply the function to create derived_contact_salutation
    df['derived_contact_purchaser_type'] = df.apply(impute_purchaser_type, axis=1)

    if df['derived_contact_purchaser_type'].isnull().any():
      null_row_number = df['derived_contact_purchaser_type'].isnull().sum()
      report_content = f'derived_contact_purchaser_type - Still contain {null_row_number} null rows after cleaning, all null rows removed'
      write_report(report_content, report_file_path)
      df = df[df['derived_contact_purchaser_type'].notnull()]

    print('Cleaned purchaser_type')
    return df

def clean_nationality(df, derived_data_dir, report_file_path):
    # Initialize 'derived_nationality' with 'customer_nationality'
    df['derived_nationality'] = df['customer_nationality']

    # Impute 'UNDEFINED' or 'OTHERS' in 'derived_nationality' with 'contact_nationality' if it is valid
    df['derived_nationality'] = df.apply(
        lambda row: row['contact_nationality'] if row['derived_nationality'].strip().upper() in ['UNDEFINED', 'OTHERS'] and row['contact_nationality'].strip().upper() not in ['UNDEFINED', 'OTHERS', '-'] else row['derived_nationality'],
        axis=1
    )

    # Path to the derived nationality CSV file
    file_path = os.path.join(derived_data_dir, 'derived_nationality.csv')

    # Check if the file path exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read the CSV file into a DataFrame
    nation_df = pd.read_csv(file_path, low_memory=False)

    # Rename the column for clarity
    nation_df.rename(columns={'derived_nationality': 'derived_nation'}, inplace=True)

    # Merge the derived nation information based on 'sales_id'
    df = df.merge(nation_df[['sales_id', 'derived_nation']], on='sales_id', how='left', suffixes=('', '_update'))

    # Impute 'Undefined' or 'OTHERS' in 'derived_nationality' with 'derived_nation'
    df['derived_nationality'] = df.apply(
        lambda row: row['derived_nation'] if row['derived_nationality'].strip().upper() in ['UNDEFINED', 'OTHERS'] else row['derived_nationality'],
        axis=1
    )

    # Drop the temporary 'derived_nation' column
    df.drop(columns=['derived_nation'], inplace=True)

    # Fill any remaining null values in 'derived_nationality' with 'OTHERS'
    df['derived_nationality'].fillna('OTHERS', inplace=True)

    if df['derived_nationality'].isnull().any():
      null_row_number = df['derived_nationality'].isnull().sum()
      report_content = f'derived_nationality - Still contain {null_row_number} null rows after cleaning, all null rows removed'
      write_report(report_content, report_file_path)
      df = df[df['derived_nationality'].notnull()]

    print('Cleaned nationality')
    return df

def clean_age(df, report_file_path):

  df['spa_date'] = pd.to_datetime(df['spa_date'])
  df['buyer_dob'] = pd.to_datetime(df['buyer_dob'])

  def calculate_derived_age(row, current_year, mean_age):
      if row['derived_contact_identification_type'] == 'REG NO.':
          return -1
      elif pd.notnull(row['buyer_dob']):
          return row['spa_date'].year - row['buyer_dob'].year
      elif pd.notnull(row['customer_age']):
          return row['customer_age']
      return mean_age


  # Calculate the current year and the mean age for imputation
  current_year = pd.Timestamp.now().year
  mean_age = df.loc[
      (df['derived_contact_identification_type'] != 'REG NO.') &
      df['customer_age'].notnull(),
      'customer_age'
  ].mean()

  # Apply the calculate_derived_age function to each row
  df['derived_age'] = df.apply(lambda row: calculate_derived_age(row, current_year, mean_age), axis=1)

  # Reimpute derived_age with the mean if the value is below 0, excluding REG NO. cases
  mask_below_zero = (df['derived_age'] <= 0) & (df['derived_contact_identification_type'] != 'REG NO.')
  df.loc[mask_below_zero, 'derived_age'] = mean_age

  if df['derived_age'].isnull().any():
    null_row_number = df['derived_age'].isnull().sum()
    report_content = f'derived_age - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_age'].notnull()]

  print('Cleaned age')
  return df


def clean_occupation(df, report_file_path):

  df['derived_contact_occupation'] = df['contact_occupation'].str.strip().str.upper()

  if df['derived_contact_occupation'].isnull().any():
    null_row_number = df['derived_contact_occupation'].isnull().sum()
    report_content = f'derived_contact_occupation - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_contact_occupation'].notnull()]

  print("Cleaned occupation")
  return df

def clean_is_cash_buyer(df, report_file_path):

  # Add a new column 'derived_is_cash_buyer' and set it to the values of 'is_cash_buyer'
  df['derived_is_cash_buyer'] = df['is_cash_buyer'].str.upper()

  # Update 'derived_is_cash_buyer' based on the given conditions
  df.loc[
       df['financier1_name'].isnull() &
      (df['is_cash_buyer'].str.strip().str.upper() == 'LOAN'),
      'derived_is_cash_buyer'] = 'CASH'

  if df['derived_is_cash_buyer'].isnull().any():
    null_row_number = df['derived_is_cash_buyer'].isnull().sum()
    report_content = f'derived_is_cash_buyer - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_is_cash_buyer'].notnull()]


  print("Cleaned is_cash_buyer")
  return df


def clean_financier(df, report_file_path):

  # Create a new column derived_financier_name and populate with financier1_name
  df['derived_financier_name'] = df['financier1_name'].str.upper()
  # Impute NaN values in derived_financier_name based on derived_is_cash_buyer
  df['derived_financier_name'] = df.apply(
      lambda row: 'CASH' if row['derived_is_cash_buyer'] == 'CASH' else 'NOT APPLICABLE' if pd.isna(row['derived_financier_name']) else row['derived_financier_name'],
      axis=1
  )

  if df['derived_financier_name'].isnull().any():
    null_row_number = df['derived_financier_name'].isnull().sum()
    report_content = f'derived_financier_name - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_financier_name'].notnull()]

  print("Cleaned financier1_name")
  return df


def clean_race(df, report_file_path):

  # Step 1: Add a new column 'derived_race' and set it to the values of 'customer_race'
  df['derived_race'] = df['customer_race'].str.upper()

  # Step 2: Update 'derived_race' with 'contact_race' where 'derived_race' is 'N/A', 'OTHERS', 'UNDEFINED'
  # and 'contact_race' is valid
  df.loc[
      df['derived_race'].str.strip().str.upper().isin(['N/A', 'OTHERS', 'UNDEFINED']) &
      ~df['contact_race'].str.strip().str.upper().isin(['N/A', 'OTHERS', 'UNDEFINED']),
      'derived_race'
  ] = df['contact_race'].str.upper()

  # Step 3: Update 'derived_race' with 'customer_race_group' where 'derived_race' is 'N/A', 'OTHERS', 'UNDEFINED'
  # and 'contact_race' is 'N/A', 'OTHERS', 'UNDEFINED' and 'customer_race_group' is valid
  df.loc[
      df['derived_race'].str.strip().str.upper().isin(['N/A', 'OTHERS', 'UNDEFINED']) &
      df['contact_race'].str.strip().str.upper().isin(['N/A', 'OTHERS', 'UNDEFINED']) &
      ~df['customer_race_group'].str.strip().str.upper().isin(['N/A', 'OTHERS', 'UNDEFINED']),
      'derived_race'
  ] = df['customer_race_group'].str.upper()


  # Replace 'N/A', 'OTHERS', 'UNDEFINED', and null values in 'derived_race' with 'OTHERS'
  df.loc[
      df['derived_race'].str.strip().str.upper().isin(['N/A', 'OTHERS', 'UNDEFINED']) |
      df['derived_race'].isnull(),
      'derived_race'
  ] = 'OTHERS'

  # Set the derived_race to 'Not Applicable' if derived_contact_identification_type is 'Reg No.'
  df.loc[df['derived_contact_identification_type'] == 'REG NO.', 'derived_race'] = 'NOT APPLICABLE'

  if df['derived_race'].isnull().any():
    null_row_number = df['derived_race'].isnull().sum()
    report_content = f'derived_race - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_race'].notnull()]

  print("Cleaned race")
  return df

def clean_contact_city(df, derived_data_dir, report_file_path):

  # Path to the derived nationality CSV file
  file_path = os.path.join(derived_data_dir, 'derived_contact_city.csv')

  # Check if the file path exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")


  # Read the CSV file into a DataFrame
  contact_city_df = pd.read_csv(file_path,low_memory=False)

  # Step 1: Create the new column 'derived_contact_city' and set it to the values of 'contact_city'
  df['derived_contact_city'] = df['contact_city']

  # Step 2: Update 'derived_contact_city' in extracted_df using 'derived_contact_city' from contact_city_df based on 'sales_id'
  df = df.merge(contact_city_df[['sales_id', 'derived_contact_city']], on='sales_id', how='left', suffixes=('', '_update'))

  # Step 3: Fill missing values in 'derived_contact_city' with 'derived_contact_city_update'
  df['derived_contact_city'] = df['derived_contact_city_update'].combine_first(df['derived_contact_city'])

  # Drop the temporary 'derived_contact_city_update' column
  df.drop(columns=['derived_contact_city_update'], inplace=True)

  df['derived_contact_city'] = df['derived_contact_city'].str.strip().str.upper()

  if df['derived_contact_city'].isnull().any():
    null_row_number = df['derived_contact_city'].isnull().sum()
    report_content = f'derived_contact_city - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_contact_city'].notnull()]

  print("Cleaned contact_city")
  return df

def clean_contact_staff(df, report_file_path):
  # Replace 'Undefined' with 'Not Staff' in the 'contact_staff' column
  df['derived_contact_staff'] = df['contact_staff'].replace('Undefined', 'Not Staff').str.upper()
  # Set the derived_contact_staff to 'Not Applicable' if derived_contact_identification_type is 'Reg No.'
  df.loc[df['derived_contact_identification_type'] == 'REG NO.', 'derived_contact_staff'] = 'NOT APPLICABLE'

  if df['derived_contact_staff'].isnull().any():
    null_row_number = df['derived_contact_staff'].isnull().sum()
    report_content = f'derived_contact_staff - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_contact_staff'].notnull()]

  print("Cleaned contact_staff")
  return df

def clean_lifestyle(df, report_file_path):

  # Replace 'Undefined' with 'Non-Registered' in the 'contact_lifestyle_registration_status' column
  df['derived_contact_lifestyle_registration_status'] = df['contact_lifestyle_registration_status'].replace('Undefined', 'Non-Registered').fillna('Non-Registered').str.upper()
  # Set the derived_contact_lifestyle_registration_status to 'Not Applicable' if derived_contact_identification_type is 'Reg No.'
  df.loc[df['derived_contact_identification_type'] == 'REG NO.', 'derived_contact_lifestyle_registration_status'] = 'NOT APPLICABLE'

  if df['derived_contact_lifestyle_registration_status'].isnull().any():
    null_row_number = df['derived_contact_lifestyle_registration_status'].isnull().sum()
    report_content = f'derived_contact_lifestyle_registration_status - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_contact_lifestyle_registration_status'].notnull()]

  print("Cleaned contact_lifestyle_registration_status")
  return df

def clean_annual_income(df, report_file_path):

  # Calculate the mode of the contact_annual_income column
  mode_income = df['contact_annual_income'].mode()[0]

  # Impute null values with the mode
  df['derived_contact_annual_income'] = df['contact_annual_income'].replace(['Not Applicable', 'Undefined','-'], mode_income).fillna(mode_income).str.upper()

  if df['derived_contact_annual_income'].isnull().any():
    null_row_number = df['derived_contact_annual_income'].isnull().sum()
    report_content = f'derived_contact_annual_income - Still contain {null_row_number} null rows after cleaning, all null rows removed'
    write_report(report_content, report_file_path)
    df = df[df['derived_contact_annual_income'].notnull()]


  print("Cleaned contact_annual_income")
  return df


def clean_non_land_posted_features(df):

  # Replace NaN values in 'non_land_posted_total_rebate' and 'total_rebate' with 0
  df['derived_non_land_posted_list_price'] = df['non_land_posted_list_price'].fillna(0)
  df['derived_non_land_posted_selling_price'] = df['non_land_posted_selling_price'].fillna(0)
  df['derived_non_land_posted_total_loan_amount'] = df['non_land_posted_total_loan_amount'].fillna(0)
  df['derived_non_land_posted_spa_amount'] = df['non_land_posted_spa_amount'].fillna(0)
  df['derived_non_land_posted_sales_amount'] = df['non_land_posted_sales_amount'].fillna(0)
  df['derived_non_land_posted_total_rebate'] = df['non_land_posted_total_rebate'].fillna(0)

  print("Cleaned non_land_posted_features")
  return df


def data_cleaning(df, config):
  print("Data cleaning in progress...")
  derived_data_dir=config['derived_data_dir']
  data_report_dir = config['data_report_dir']
  save_dir = config['save_dir']
  save_csv= config['save_output']

  report_file_path = os.path.join(data_report_dir, 'clean_data_report.txt')

  if derived_data_dir == None:
      raise ValueError(f"Derived data directory is missing")

  # Clean customer features
  df = clean_id_type(df, derived_data_dir, report_file_path)
  df = clean_bumi_status(df, report_file_path)
  df = clean_gender(df, report_file_path)
  df = clean_marital_status(df, report_file_path)
  df = clean_salutation(df, report_file_path)
  df = clean_purchaser_type(df, report_file_path)
  df = clean_nationality(df, derived_data_dir, report_file_path)
  df = clean_age(df, report_file_path)
  df = clean_occupation(df, report_file_path)
  df = clean_is_cash_buyer(df, report_file_path)
  df = clean_race(df, report_file_path)
  df = clean_contact_city(df, derived_data_dir, report_file_path)
  df = clean_contact_staff(df, report_file_path)
  df = clean_lifestyle(df, report_file_path)
  df = clean_annual_income(df, report_file_path)

   # Clean property features
  df = clean_built_up_area_sqft(df, derived_data_dir, report_file_path)
  df = clean_build_type(df, report_file_path)
  df = clean_phase_property_type(df, report_file_path)
  df = clean_financier(df, report_file_path)
  df = clean_non_land_posted_features(df)

  print(f"Data cleaning completed!")
  if save_csv:
        if save_dir is None:
            raise ValueError("save_dir must be provided if save_csv is True.")

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the resulting DataFrame to CSV
        output_csv_path = os.path.join(save_dir, 'clean_df.csv')
        df.to_csv(output_csv_path, index=False)
        print(f"Data cleaning and saving completed! Saved to {output_csv_path}")
    
  
  return df
