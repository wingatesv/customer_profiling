import os
import warnings
import pandas as pd
from tqdm import tqdm


def read_columns_to_extract(columns_file):
    """
    Reads the columns to extract from the CSV file.
    
    Args:
    columns_file (str): Path to the CSV file containing the column names to extract.
    
    Returns:
    list: A list of column names to extract.
    """
    columns_to_extract_df = pd.read_csv(columns_file)
    return columns_to_extract_df['column_name'].tolist()

def check_columns_presence(df, required_columns, context=""):
    """
    Checks for the presence of required columns in the DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame to check.
    required_columns (list): A list of required column names.
    context (str): A string indicating the context for the check (for error messages).
    
    Raises:
    ValueError: If any required columns are missing from the DataFrame.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame {context}: {missing_columns}")

def convert_date_columns(df, date_columns):
    """
    Converts specified columns to datetime.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the columns to convert.
    date_columns (list): A list of column names to convert to datetime.
    """
    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')

def filter_dataframe(df, filter_conditions):
    """
    Filters the DataFrame based on provided conditions.
    
    Args:
    df (pd.DataFrame): The DataFrame to filter.
    filter_conditions (pd.Series): A boolean Series indicating the filter conditions.
    
    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    return df[filter_conditions]

def extract_columns(df, columns_to_extract):
    """
    Extracts specified columns from the DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame to extract columns from.
    columns_to_extract (list): A list of column names to extract.
    
    Returns:
    pd.DataFrame: The DataFrame containing only the specified columns.
    """
    return df.loc[:, columns_to_extract]

def remove_recent_transaction(df, config):
    """
    Removes rows with transactions within one month for each config['unique_customer_id'].
    
    Args:
    df (pd.DataFrame): The DataFrame containing the transaction data.
    
    Returns:
    pd.DataFrame: The DataFrame with recent transactions removed.
    """
    # Sort the DataFrame by config['unique_customer_id'] and 'spa_date'
    df = df.sort_values(by=[config['unique_customer_id'], 'spa_date'])
    # Track the number of excluded rows
    # excluded_count = 0
    # Initialize an empty DataFrame to collect the rows that meet the condition
    result_df = pd.DataFrame()

    # Iterate over each config['unique_customer_id'] group
    for _, group in tqdm(df.groupby(config['unique_customer_id']), desc="Checking the recency of the rows"):
        # Initialize a list to keep track of rows to retain
        to_retain = []


        # Iterate through the group to find rows to retain
        i = 0
        while i < len(group):
            current_row = group.iloc[i]
            # Find the range of rows with transaction dates within 30 days
            j = i
            while j + 1 < len(group) and (group.iloc[j + 1]['spa_date'] - current_row['spa_date']).days <= 30:
                j += 1

            # Retain the last row in the range
            to_retain.append(group.index[j])
            # Move to the next range
            i = j + 1

        # Calculate the number of excluded rows for this group
        # excluded_count += len(group) - len(to_retain)
        # Append the rows to retain to the result DataFrame
        result_df = pd.concat([result_df, group.loc[to_retain]])
        # print(f"Total rows excluded: {excluded_count}")

    return result_df

def data_extraction(df, config):
    """Performs data extraction and returns the resulting DataFrame."""
    print("Data extraction in progress...")

    columns_file= config['data_extraction_columns_file'] 
    save_dir= config['save_dir']
    save_csv= config['save_output']

    # Read the columns to extract from the CSV file
    columns_to_extract = read_columns_to_extract(columns_file)

    # List of columns to convert to datetime
    date_columns = [
        'transaction_date', 'commencement_date', 'completion_date',
        'spa_stamp_date', 'sales_conversion_date', 'spa_date',
        'contact_reg_date', 'buyer_dob'
    ]

    # Check for the presence of date columns
    check_columns_presence(df, date_columns, "for date conversion")

    # Convert date columns to datetime
    convert_date_columns(df, date_columns)


    # Check for the presence of filter columns
    filter_columns = ['project_country', 'last_sales_status', 'date_type']
    check_columns_presence(df, filter_columns, "for filtering")

    # Filter the DataFrame based on specific conditions for df_temp_1
    filter_conditions_1 = (
        (df['project_country'] == 'Malaysia') &
        (df['last_sales_status'] == 'Sold') &
        (df['date_type'] == 'spa_date_type')
    )
    df_temp_1 = filter_dataframe(df, filter_conditions_1)

    # Check for the presence of columns to extract in df_temp_1
    check_columns_presence(df_temp_1, columns_to_extract, "for extraction in df_temp_1")

    # Extract the specified columns for df_temp_1
    df_temp_1 = extract_columns(df_temp_1, columns_to_extract)

    # Define additional columns to extract for df_temp_2
    additional_columns_to_extract = [
        'sales_id',
        'non_land_posted_list_price', 'non_land_posted_selling_price', 'non_land_posted_net_selling_price',
        'non_land_posted_total_rebate', 'non_land_posted_total_loan_amount', 'non_land_posted_spa_amount', 'non_land_posted_sales_amount',
    ]

    # Check for the presence of additional columns
    check_columns_presence(df, additional_columns_to_extract, "for extraction in df_temp_2")

    # Apply the initial filters to df_temp_2
    filter_conditions_2 = (
        (df['project_country'] == 'Malaysia') &
        (df['last_sales_status'] == 'Sold') &
        (df['date_type'] == 'spa_month_type')
    )
    df_temp_2 = filter_dataframe(df, filter_conditions_2)

    # Extract the specified additional columns for df_temp_2
    df_temp_2 = extract_columns(df_temp_2, additional_columns_to_extract)

    # Merge DataFrames on specified columns
    output_df = pd.merge(df_temp_1, df_temp_2, on=['sales_id'], how='inner')

    # Remove rows with transactions within one month
    output_df = remove_recent_transaction(output_df, config)

    oldest_date = output_df['spa_date'].min()
    latest_date = output_df['spa_date'].max()
    print(f"Using data from {oldest_date} to {latest_date}...")
    
    
    print(f"Using data from {config['evaluation_start_date']} to {config['evaluation_end_date']} for evaluation")
    # Extract start and end dates from the config
    start_date = pd.to_datetime(config['evaluation_start_date'], errors='coerce')
    # Check for end date, if not available or out of range, use the latest date from spa_date and raise a warning
    end_date = pd.to_datetime(config['evaluation_end_date'], errors='coerce')

    # Validate dates within the range of the DataFrame
    if start_date < output_df['spa_date'].min() or start_date > output_df['spa_date'].max():
        max_year = output_df['spa_date'].max().year
        start_date = pd.Timestamp(year=max_year - 1, month=1, day=1)
        warnings.warn(f"evaluation_start_date {config['evaluation_start_date']} is out of range, using the latest second year date from spa_date: {start_date}")

            
    if end_date < output_df['spa_date'].min() or end_date > output_df['spa_date'].max():
        end_date = output_df['spa_date'].max()
        warnings.warn(f"evaluation_end_date {config['evaluation_end_date']} is out of range, using the latest date from spa_date: {end_date}")

    # Split the DataFrame based on the validated start and end dates
    split_eval_df = output_df[(output_df['spa_date'] >= start_date) & (output_df['spa_date'] <= end_date)]
    output_df = output_df[output_df['spa_date'] < start_date]
    
    # Save the split dataframes to CSV files
    output_csv_path = os.path.join(save_dir, 'for_eval_label_generation.csv')
    split_eval_df.to_csv(output_csv_path, index=False)

    print(f"Using data from {start_date} to {end_date} for evaluation")
    print(f"Evaluation data saved to {output_csv_path}")

    # Check for data after the end_date
    after_end_date_df = output_df[output_df['spa_date'] > end_date]
    if not after_end_date_df.empty:
        print(f"Data after {end_date} exists but will be appended back to the training df")
        print(f"Number of rows appended: {len(after_end_date_df)}")

        # Append after_end_date_df back to output_df
        output_df = pd.concat([output_df, after_end_date_df])
            

    print("Data extraction phase completed!")

    if save_csv:
        if save_dir is None:
            raise ValueError("save_dir must be provided if save_csv is True.")
        
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the resulting DataFrame to CSV
        output_csv_path = os.path.join(save_dir, 'data_extraction_df.csv')
        output_df.to_csv(output_csv_path, index=False)
        print(f"Data extraction and saving completed! Saved to {output_csv_path}")

    return output_df
