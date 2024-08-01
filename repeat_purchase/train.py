import pandas as pd
import numpy as np
import subprocess
import sys
import joblib
# import lightgbm as lgb
import os
import warnings
import shutil
# import catboost as cb
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

from data.data_extraction import read_columns_to_extract, check_columns_presence, extract_columns


def encode_features(df):
    # Drop the 'label' column
    df = df.drop(columns=['label'], errors='ignore')
    # Identify categorical and numerical columns
    categorical_features = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Define the transformers for numerical and categorical features
    numerical_transformer = SimpleImputer(strategy='mean')

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the training data
    train_encoded = preprocessor.fit_transform(df)

    return train_encoded, preprocessor, categorical_features, numerical_features




def bootstrap_aggregation(df, num_splits):
    if num_splits > 5:
        warnings.warn("The number of splits cannot be greater than 5. Setting num_splits to 5.")
        num_splits = 5
    print("Implementing bootstrap aggregation....")
    df['spa_date'] = pd.to_datetime(df['spa_date'])
    df['spa_year'] = df['spa_date'].dt.year

    # Define the split year
    split_year = 2015

    if num_splits == 1:
        # Use all years if num_splits is 1
        bootstrap_datasets = [df]
        print(f"Single Split: Start Year = {df['spa_year'].min()}, End Year = {df['spa_year'].max()}")
    else:
        # Adjust num_splits to account for the final split
        num_splits -= 1

        # Get the minimum and maximum years
        min_year = df['spa_year'].min()
        max_year = split_year - 1

        # Calculate the range and the step for the splits
        year_range = max_year - min_year + 1
        step = year_range // num_splits

        bootstrap_datasets = []

        # Generate the splits
        for i in range(num_splits):
            start_year = min_year + i * step
            end_year = start_year + step - 1
            if i == num_splits - 1:  # Ensure the last split goes up to max_year
                end_year = max_year
            split_df = df[(df['spa_year'] >= start_year) & (df['spa_year'] <= end_year)]
            bootstrap_datasets.append(split_df)
            print(f"Split {i + 1}: Start Year = {start_year}, End Year = {end_year}")

        # Include the years from 2015 to the latest year
        final_split_df = df[df['spa_year'] >= split_year]
        bootstrap_datasets.append(final_split_df)
        print(f"Final Split: Start Year = {split_year}, End Year = {df['spa_year'].max()}")

    return bootstrap_datasets


# Function to downsample a single dataset
def downsample_df(df, ratio):
    # Separate majority and minority classes
    df_majority = df[df['label'] == 0]
    df_minority = df[df['label'] == 1]

    # Undersample the majority class
    df_majority_undersampled = resample(df_majority,
                                        replace=False,    # sample without replacement
                                        n_samples=len(df_minority)*ratio, # to match minority class
                                        random_state=123) # reproducible results

    # Combine minority class with undersampled majority class
    df_balanced = pd.concat([df_majority_undersampled, df_minority])

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=123).reset_index(drop=True)

    return df_balanced

def get_feature_importances(model, X_train, preprocessor, categorical_features, numerical_features, rp_model):
    # Get feature names from the preprocessor
    ohe = preprocessor.named_transformers_['cat']['onehot']
    feature_names = numerical_features + list(ohe.get_feature_names_out(categorical_features))

    if rp_model == "mlp":
      X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train

    # Check for feature importance in specific models
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_ , feature_names
    elif hasattr(model, 'coef_'):
        return np.abs(model.coef_[0]), feature_names
    else:
        # Use permutation importance for models without built-in feature importance
        result = permutation_importance(model, X_train, model.predict(X_train), n_repeats=10, random_state=42)
        return result.importances_mean, feature_names

def model_training(df_list, config, preprocessor, categorical_features, numerical_features):

  for i, bootstrap_df in enumerate(df_list):
      # Perform train-validation split with 20% of the train set used as validation set and fixed seed
      train_df, val_df = train_test_split(bootstrap_df, test_size=config['rp_validation_split'], random_state=42)

      # Separate the target variable
      train_target = train_df['label']
      val_target = val_df['label']

      # Drop the target variable from the feature set
      train_df_2 = train_df.drop(columns=['label', 'spa_date'], errors='ignore')
      val_df_2 = val_df.drop(columns=['label', 'spa_date'], errors='ignore')

      # Use the fitted preprocessor to transform the data
      train_encoded = preprocessor.transform(train_df_2)
      val_encoded = preprocessor.transform(val_df_2)

      # Automatically compute class weights
      class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=train_target)
      class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

      # Train and evaluate the model
      model, report = train_model(config['rp_model'], train_encoded, train_target, val_encoded, val_target, class_weight=class_weight_dict)
      
      print(f"Model: {config['rp_model']}")
      print("Validation Classification Report:")
      print(report)

      output_model_path = os.path.join(config['rp_model_dir'], f"{config['rp_model']}_{(i+1)}.pkl")
      joblib.dump(model, output_model_path)
      print(f"Save model in {output_model_path}...")

      # Get feature importances if possible
      feature_importances, feature_names = get_feature_importances(model, train_encoded, preprocessor, categorical_features, numerical_features, config['rp_model'])

      # Create a DataFrame for better visualization
      feature_importances_df = pd.DataFrame({
          'Feature': feature_names,
          'Importance': feature_importances
      })
      feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
      output_csv_path = os.path.join(config['rp_model_dir'], f'feature_importance_{(i+1)}.csv')
      feature_importances_df.to_csv(output_csv_path, index=False)
      print(f"Model prediction saved to {output_csv_path}")

 
def train_model(model_name, train_encoded, train_target, test_encoded, test_target, class_weight=None):
    models = {
        'logistic_regression': LogisticRegression(class_weight=class_weight),
        'knn': KNeighborsClassifier(),  # KNN does not support class_weight directly
        'svm': SVC(class_weight=class_weight),
        'decision_tree': DecisionTreeClassifier(class_weight=class_weight),
        'random_forest': RandomForestClassifier(class_weight=class_weight),
        'gradient_boosting': GradientBoostingClassifier(),  # Gradient Boosting does not support class_weight directly
        # 'lightgbm': lgb.LGBMClassifier(class_weight=class_weight),
        # 'catboost': cb.CatBoostClassifier(verbose=0, class_weights=class_weight),  # CatBoost uses class_weights parameter
         'mlp': MLPClassifier(
                              hidden_layer_sizes=(64, 32),  # two hidden layers
                              activation='relu',          # ReLU activation function
                              solver='adam',              # Optimizer
                              alpha=0.0001,               # L2 regularization term (default)
                              batch_size=16,              # Batch size
                              learning_rate='adaptive',   # Learning rate adjustment
                              learning_rate_init=0.001,   # Initial learning rate
                              max_iter=1000,               # Maximum number of iterations
                              random_state=42,            # Seed for reproducibility
                              early_stopping=True,        # Enable early stopping
                              validation_fraction=0.15,    # Fraction of training data to set aside for validation
                              n_iter_no_change=10         # Number of iterations with no improvement to wait before stopping
                          )
            }

    if model_name not in models:
        raise ValueError(f"Model {model_name} is not supported. Choose from {list(models.keys())}")

    model = models[model_name]

    # Train the model
    model.fit(train_encoded, train_target)

    # Predict on the test data
    test_predictions = model.predict(test_encoded)

    # Convert to consistent type
    test_target = test_target.astype(int)
    test_predictions = test_predictions.astype(int)

    # Evaluate the model using classification report
    report = classification_report(test_target, test_predictions)

    return model, report


def train(df, config):
  if os.path.exists(config['rp_model_dir']):
    # Remove all files in the directory
    print("Found existing directory...delete all files for model training")
    for filename in os.listdir(config['rp_model_dir']):
        file_path = os.path.join(config['rp_model_dir'], filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
  else:
      os.makedirs(config['rp_model_dir'])
      print(f"Created directory: {config['rp_model_dir']}")


  # Read the columns to extract from the CSV file
  columns_to_extract = read_columns_to_extract(config["rp_model_columns_file"])
  columns_to_extract.append(config['unique_customer_id'])
  # Check for the presence of columns to extract in df
  check_columns_presence(df, columns_to_extract, "for rp_model extraction")
  # Extract the specified columns for df_temp_1
  df = extract_columns(df, columns_to_extract)



  # See the distribution of the label column
  label_distribution = df['label'].value_counts()
  print("Distribution of the training label:")
  print(label_distribution)


  # Encode features before bootstraping
  df_temp = df.drop(columns=['label', 'spa_date', config['unique_customer_id']], errors='ignore')
  _, preprocessor, categorical_features, numerical_features = encode_features(df_temp)
  output_preprocessor_path = os.path.join(config['rp_model_dir'], 'preprocessor.pkl')
  joblib.dump(preprocessor, output_preprocessor_path)
  print(f"Save preprocessor in {output_preprocessor_path}...")

  # Perform bootstraping
  df_list = bootstrap_aggregation(df=df, num_splits= config['bagging_split'])

  # Perform downsampling if condition True
  if config['down_sampling']:
      print("Performing down sampling...")
      df_list = [downsample_df(df, config['down_sampling_ratio']) for df in df_list]
      print(f"Each split is down sampled with 1:{config['down_sampling_ratio']} ratio with the majority class")

  model_training(df_list=df_list, config=config, preprocessor=preprocessor, categorical_features=categorical_features, numerical_features=numerical_features)
  

