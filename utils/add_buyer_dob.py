import pandas as pd
from google.cloud import bigquery


def main():

    # Set up the BigQuery client
    client = bigquery.Client(project='gl-data-warehouse-prototype')  # Replace with your project ID

    # Define the SQL query to add a new column and update it
    sql_query = """
    -- Step 1: Add a new column 'buyer_dob' to the 'metric_nmkt_fact_sales' table
    ALTER TABLE `gl-data-warehouse-prototype.quandatics_ml.metric_nmkt_fact_sales`
    ADD COLUMN buyer_dob DATETIME;

    -- Step 2: Update the 'buyer_dob' column by joining with the 'rr_buyer_details' table
    UPDATE `gl-data-warehouse-prototype.quandatics_ml.metric_nmkt_fact_sales` AS sales
    SET buyer_dob = (
      SELECT buyer_dob
      FROM `gl-data-warehouse-prototype.quandatics_ml.rr_buyer_details` AS details
      WHERE sales.sales_id = details.sales_id
    )
    WHERE sales.sales_id IN (
      SELECT sales_id
      FROM `gl-data-warehouse-prototype.quandatics_ml.rr_buyer_details`
    );
    """

    # Execute the SQL query
    query_job = client.query(sql_query)

    # Wait for the query to complete
    query_job.result()
    print("Query executed successfully.")

    # If you want to download the table data to a Pandas DataFrame
    table_id = "gl-data-warehouse-prototype.quandatics_ml.metric_nmkt_fact_sales"

    # Query to select all data from the table
    df_query = f"SELECT * FROM `{table_id}`"

    # Execute the query and download the data to a DataFrame
    results = client.query(df_query).to_dataframe()

    # Display the DataFrame
    print(results.head())

if __name__ == "__main__":
    main()
