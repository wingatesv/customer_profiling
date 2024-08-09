# Customer Profiling

## Overview

This repository contains the codebase for preparing data, training machine learning models, and evaluating models for customer profile prediction. The project is specifically focused on predicting customer behaviors such as repeat purchases and property type preferences using various machine learning algorithms.

## How to Run the Code

### Prerequisites

The code has been tested on the Google Colab platform. Make sure you have access to Google Colab and a Google Drive account to store and access your data.

### Clone the Repository

First, clone the repository to your Google Colab environment:

```bash
!git clone https://github.com/wingatesv/customer_profiling.git
```
#### Install Required Packages
Some additional Python packages may be required. You can install these packages using the following command:

```bash
!pip install -r /content/customer_profiling/requirement.txt
```

### Configure the config.yaml File
Before running the main script, configure the config.yaml file according to your project needs. Below is a detailed explanation of the parameters available in the config.yaml file.

### Run the Main Script
To run the main script, use the following command in your Google Colab environment:

```bash
!python /content/customer_profiling/main.py --config /content/customer_profiling/config.yaml
```
## Configuration Parameters (config.yaml)
### Main Section
generate_test_label_mode: Set to True to generate test labels for the input file only. Defaults to False.

training_mode: Set to True to enable training mode. Defaults to False.

evaluation_mode: Set to True to enable evaluation mode. Defaults to False.

inference_mode: Set to True to enable inference mode. Defaults to False.

### Model Prediction Modes
repeat_purchase_mode: Set to True to enable repeat purchase modeling. Defaults to True.

property_type_mode: Set to True to enable property type modeling. Defaults to True.

unique_customer_id: Specify the unique identifier for customers, e.g., contact_nric_masked.

### Data Preparation Section
save_output: Set to True to save the output of the data preparation process. Defaults to True.

load_df_from_checkpoint: Set to True to load the DataFrame from a checkpoint. Defaults to False.

split_df_for_eval: Set to True if you want to split the DataFrame for evaluation. Defaults to False.

input_file: Path to the input raw data file. Example: /content/drive/MyDrive/eval_data/raw_df/raw_df.csv.

data_extraction_columns_file: Path to the CSV file that contains the columns to extract for data preparation. Example: /content/customer_profiling/columns/columns_to_extract.csv.

feature_mapping_dir: Directory containing feature mappings. Example: /content/customer_profiling/feature_mapping/.

derived_data_dir: Directory to save derived data. Example: /content/customer_profiling/derived_data/.

data_report_dir: Directory to save data reports. Example: /content/customer_profiling/data_report/.

save_dir: Directory to save the final output. Example: /content/output/.

### Test Label Generation Section
evaluation_start_date: Start date for evaluation in yyyy-mm-dd format. Example: 2024-01-01.

evaluation_end_date: End date for evaluation in yyyy-mm-dd format. Example: 2024-07-31.

### Repeat Purchase Mode Section
rp_model: Specify the model type for repeat purchase prediction. 

rp_model_dir: Directory for the repeat purchase model. Example: /content/customer_profiling/models/rp_model/.

rp_model_columns_file: Path to the columns file for the repeat purchase model. Example: /content/customer_profiling/columns/rp_model_columns_to_extract.csv.

bagging_split: Number of bagging splits for training. Defaults to 3.

down_sampling: Set to True to enable down sampling. Defaults to True.

down_sampling_ratio: Ratio for down sampling. Defaults to 1.

rp_validation_split: Validation split ratio. Defaults to 0.2.

inference_threshold: Threshold for inference. Defaults to 0.6.

rp_result_dir: Directory to save repeat purchase model results. Example: /content/1_rp_result/.

### Property Type Mode Section
pt_model: Specify the model type for property type prediction. Possible values include:

pt_model_dir: Directory for the property type model. Example: /content/customer_profiling/models/pt_model/.

pt_model_columns_file: Path to the columns file for the property type model. Example: /content/customer_profiling/columns/pt_model_columns_to_extract.csv.

pt_validation_split: Validation split ratio. Defaults to 0.2.

landed_mode: Set to True to enable landed property type mode. Defaults to True.

high_rise_mode: Set to True to enable high-rise property type mode. Defaults to True.

commercial_mode: Set to True to enable commercial property type mode. Defaults to True.

pt_result_dir: Directory to save property type model results. Example: /content/2_pt_result/.

## Contributing
If you would like to contribute to this project, please feel free to create a pull request or submit an issue on GitHub.
