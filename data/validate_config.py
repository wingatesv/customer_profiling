import os
import sys
import re

def validate_config(config):
    # List of valid model types
    valid_models = [
        "logistic_regression", "knn", "svm", "decision_tree",
        "random_forest", "gradient_boosting", "mlp"
    ]
    
    # Date format regex
    date_format = re.compile(r"\d{4}-\d{2}-\d{2}")

    # Check main modes
    if config.get("generate_test_label_mode"):
        if config.get("training_mode") or config.get("evaluation_mode") or config.get("inference_mode"):
            print("Error: 'generate_test_label_mode' is True. 'training_mode', 'evaluation_mode', and 'inference_mode' should all be False.")
            exit()
    else:
        if not (config.get("training_mode") or config.get("evaluation_mode") or config.get("inference_mode")):
            print("Error: At least one of 'training_mode', 'evaluation_mode', or 'inference_mode' must be True.")
            exit()

    # Check model prediction modes
    if not config.get("repeat_purchase_mode") and not config.get("property_type_mode"):
        print("Error: At least one of 'repeat_purchase_mode' or 'property_type_mode' must be True.")
        exit()

    # Check data preparation
    required_paths = {
        "input_file": config.get("input_file"),
        "data_extraction_columns_file": config.get("data_extraction_columns_file"),
        "feature_mapping_dir": config.get("feature_mapping_dir"),
        "derived_data_dir": config.get("derived_data_dir"),

    }

    for path_name, path in required_paths.items():
        if not path:
            print(f"Error: '{path_name}' is not specified.")
            exit()
        if not os.path.exists(path):
            print(f"Error: '{path_name}' does not exist. Provided path: {path}")
            exit()

    # Check test label generation section
    if config.get("generate_test_label_mode"):
        if not (config.get("evaluation_start_date") and config.get("evaluation_end_date")):
            print("Error: 'evaluation_start_date' and 'evaluation_end_date' must be specified when 'generate_test_label_mode' is True.")
            exit()
        if not date_format.match(str(config.get("evaluation_start_date"))):
            print("Error: 'evaluation_start_date' must be in 'YYYY-MM-DD' format.")
            exit()
        if not date_format.match(str(config.get("evaluation_end_date"))):
            print("Error: 'evaluation_end_date' must be in 'YYYY-MM-DD' format.")
            exit()
    
    # Check repeat purchase mode section
    if config.get("repeat_purchase_mode"):
        if config.get("rp_model") not in valid_models:
            print(f"Error: 'rp_model' must be one of {valid_models}.")
            exit()
  
        if not config.get("rp_model_columns_file"):
            print("Error: 'rp_model_columns_file' is not specified.")
            exit()
        if not (1 <= config.get("bagging_split", 1) <= 5):
            print("Error: 'bagging_split' must be between 1 and 5.")
            exit()
        if not (1 <= config.get("down_sampling_ratio", 1) <= 5):
            print("Error: 'down_sampling_ratio' must be between 1 and 5.")
            exit()
        if not (0 <= config.get("rp_validation_split", 0) <= 0.5):
            print("Error: 'rp_validation_split' must be between 0 and 0.5.")
            exit()
        if not (0 <= config.get("inference_threshold", 0) <= 1):
            print("Error: 'inference_threshold' must be between 0 and 1.")
            exit()
      
    
    # Check property type mode section
    if config.get("property_type_mode"):
        if config.get("pt_model") not in valid_models:
            print(f"Error: 'pt_model' must be one of {valid_models}.")
            exit()
        if not config.get("pt_model_columns_file"):
            print("Error: 'pt_model_columns_file' is not specified.")
            exit()
        if not (0 <= config.get("pt_validation_split", 0) <= 0.5):
            print("Error: 'pt_validation_split' must be between 0 and 0.5.")
            exit()

    # Validate unique_customer_id
    if config.get("unique_customer_id") not in ["contact_nric_masked", "contact_nric"]:
        print("Error: 'unique_customer_id' must be either 'contact_nric_masked' or 'contact_nric'.")
        exit()


    # If all checks passed
    print("All configurations are valid.... continue....")
