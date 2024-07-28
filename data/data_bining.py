import os
import numpy as np
import pandas as pd


def bin_built_up_area(df):
  
  # Calculate equal bins using np.percentile, dropping duplicates
  equal_bins = np.percentile(df['derived_built_up_area_sqft'].dropna(), np.linspace(0, 100, 21), interpolation='midpoint')
  equal_bins = np.unique(equal_bins)  # Ensure bin edges are unique

  # Define labels for the bins
  labels = [f'{int(equal_bins[i])}-{int(equal_bins[i+1])} SQFT' for i in range(len(equal_bins) - 1)]
  # print(labels)

  # Convert derived_built_up_area_sqft to category using the determined bins
  # df['derived_built_up_area_sqft_category'] = pd.cut(df['derived_built_up_area_sqft'], bins=equal_bins, labels=labels, include_lowest=True)
  df.loc[:, 'derived_built_up_area_sqft_category'] = pd.cut(df['derived_built_up_area_sqft'], bins=equal_bins, labels=labels, include_lowest=True)
  


  print("Done bining built_up_area")
  return df

def bin_derived_age(df):

  equal_bins = np.percentile(df['derived_age'].dropna(), np.linspace(0, 100, 6))
  equal_labels = [f'{int(equal_bins[i])}-{int(equal_bins[i+1])} AGE-GROUP' for i in range(len(equal_bins) - 1)]
  # print(equal_labels)

  # Categorize the 'derived_age' column using Equal-Frequency Binning
  # df['derived_age_equal_category'] = pd.cut(df['derived_age'], bins=equal_bins, labels=equal_labels, include_lowest=True)
  df.loc[:, 'derived_age_equal_category'] = pd.cut(df['derived_age'], bins=equal_bins, labels=equal_labels, include_lowest=True)
  df['derived_age_equal_category'] = df['derived_age_equal_category'].cat.add_categories(['NOT APPLICABLE']).fillna('NOT APPLICABLE')

  print("Done bining derived_age")
  return df

def bin_derived_rebate_percentage(df):

  # Create a new column derived_rebate_percentage_category based on the given rules
  def rebate_category(row):
      if row['derived_rebate_percentage'] == 0:
          return "NO REBATE"
      elif 0 < row['derived_rebate_percentage'] < 0.1:
          return "< 10% REBATE"
      elif row['derived_rebate_percentage'] >= 0.1:
          return "> 10% REBATE"

  df['derived_rebate_percentage_category'] = df.apply(rebate_category, axis=1)


  print("Done bining derived_rebate_percentage")
  return df

def bin_non_land_posted_features(df):

  columns_to_plot = [
    'derived_non_land_posted_list_price', 'derived_non_land_posted_selling_price',
    'derived_non_land_posted_total_rebate', 'derived_non_land_posted_total_loan_amount', 'derived_non_land_posted_spa_amount']

  bins_labels_dict = {}
  # Determine the bins and labels for each column using equal-frequency binning
  for column in columns_to_plot:
      data = df[column].dropna()
      bins = np.percentile(data, np.linspace(0, 100, 6))  # 5 bins mean 6 percentile points
      bins = np.unique(bins)  # Ensure bin edges are unique
      labels = [f'{int(bins[i])}-{int(bins[i+1])}' for i in range(len(bins) - 1)]
      bins_labels_dict[column] = (bins, labels)


  # Convert each column to category using the determined bins and labels
  for column, (bins, labels) in bins_labels_dict.items():
      # Ensure labels are unique
      unique_labels = [f'{int(bins[i])}-{int(bins[i+1])} ({i+1})' for i in range(len(bins) - 1)]
      # df[f'{column}_category'] = pd.cut(df[column], bins=bins, labels=unique_labels, include_lowest=True)
      df.loc[:, f'{column}_category'] = pd.cut(df[column], bins=bins, labels=unique_labels, include_lowest=True)
      


  print("Done bining non_land_posted_features")
  return df


def data_bining(df, config):
  print("Data bining in progress...")
  save_dir=config['save_dir']
  save_csv=config['save_output']
  #Drop sales_id that is near 2023, 0 for the non_land_posted features
  sales_ids_to_drop =  [318456,216039]
  # Drop rows where sales_id is in the list
  df = df[~df['sales_id'].isin(sales_ids_to_drop)]
  df = df.copy()
  df = bin_built_up_area(df)
  df = bin_derived_age(df)
  df = bin_derived_rebate_percentage(df)
  df = bin_non_land_posted_features(df)

  print("Data bining phase completed!")
  if save_csv:
      if save_dir is None:
          raise ValueError("save_dir must be provided if save_csv is True.")
      
      # Ensure the save directory exists
      os.makedirs(save_dir, exist_ok=True)

      # Save the resulting DataFrame to CSV
      output_csv_path = os.path.join(save_dir, 'bin_df.csv')
      df.to_csv(output_csv_path, index=False)
      print(f"Data bining and saving completed! Saved to {output_csv_path}")

  return df
