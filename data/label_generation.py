import pandas as pd
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

def generate_label(df, save_dir=None, save_csv=False):
    # Convert contact_nric_masked to string using .loc
    df.loc[:, 'contact_nric_masked'] = df['contact_nric_masked'].astype(str)
    
    # Convert  date to datetime
    df['spa_stamp_date'] = pd.to_datetime(df['spa_stamp_date'], errors='coerce')
    df['spa_date'] = pd.to_datetime(df['spa_date'], errors='coerce')  

    # DataFrame to store results
    repeated_purchase_df = pd.DataFrame()

    # Group by customer
    grouped = df.groupby('contact_nric_masked')

    # Progress bar for processing each group
    for name, group in tqdm(grouped, desc="Generating training labels"):
        # Sort the group by transaction_date
        group = group.sort_values(by='spa_date')
        transaction_count = len(group)

        if transaction_count == 1:
            row = group.iloc[0]
            row_copy = row.copy()
            row_copy['label'] = 'N'
            row_copy['repeat_sales_id'] = 'NOT APPLICABLE'
            row_copy['repeat_phase_property_type'] = 'NOT APPLICABLE'
            row_copy['repeat_non_land_posted_selling_price'] = np.nan
            row_copy['repeat_spa_stamp_date'] = pd.Timestamp('2022-12-31').date()
            row_copy['repeat_spa_date'] = pd.Timestamp('2022-12-31').date()
            repeated_purchase_df = pd.concat([repeated_purchase_df, row_copy.to_frame().T])
        else:
            # Handle first transaction
            first_row = group.iloc[0].copy()
            first_row['label'] = 'Y'
            first_row['repeat_sales_id'] = group.iloc[1]['sales_id']
            first_row['repeat_phase_property_type'] = group.iloc[1]['phase_property_type']
            first_row['repeat_non_land_posted_selling_price'] = group.iloc[1]['non_land_posted_selling_price']
            first_row['repeat_spa_stamp_date'] = group.iloc[1]['spa_stamp_date'].date()
            first_row['repeat_spa_date'] = group.iloc[1]['spa_date'].date()
            repeated_purchase_df = pd.concat([repeated_purchase_df, first_row.to_frame().T])

            # Handle middle transactions
            for i in range(1, transaction_count - 1):
                row = group.iloc[i].copy()
                row['label'] = 'Y'
                row['repeat_sales_id'] = group.iloc[i + 1]['sales_id']
                row['repeat_phase_property_type'] = group.iloc[i + 1]['phase_property_type']
                row['repeat_non_land_posted_selling_price'] = group.iloc[i + 1]['non_land_posted_selling_price']
                row['repeat_spa_stamp_date'] = group.iloc[i + 1]['spa_stamp_date'].date()
                row['repeat_spa_date'] = group.iloc[i + 1]['spa_date'].date()
                repeated_purchase_df = pd.concat([repeated_purchase_df, row.to_frame().T])

            # Handle last transaction
            last_row = group.iloc[-1].copy()
            last_row['label'] = 'N'
            last_row['repeat_sales_id'] = 'NOT APPLICABLE'
            last_row['repeat_phase_property_type'] = 'NOT APPLICABLE'
            last_row['repeat_non_land_posted_selling_price'] = np.nan
            last_row['repeat_spa_stamp_date'] = pd.Timestamp('2022-12-31').date()
            last_row['repeat_spa_date'] = pd.Timestamp('2022-12-31').date()
            repeated_purchase_df = pd.concat([repeated_purchase_df, last_row.to_frame().T])

    print("Training label generation completed!")

    if save_csv:
        if save_dir is None:
            raise ValueError("save_dir must be provided if save_csv is True.")

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the resulting DataFrame to CSV
        output_csv_path = os.path.join(save_dir, 'label_df.csv')
        repeated_purchase_df.to_csv(output_csv_path, index=False)
        print(f"Training label generation and saving completed! Saved to {output_csv_path}")

    return repeated_purchase_df
