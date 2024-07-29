import pandas as pd

def read_csv(file_path):
    return pd.read_csv(file_path)

def add_new_row(df):
    clean_value = input("Please input clean value: ").strip().upper()
    map_value = input("Please input map value: ").strip().upper()

    new_row = df.iloc[0].copy()  # Copy the structure of the first row
    new_row[1] = clean_value
    new_row[3] = map_value

    # Create a DataFrame for the new row
    new_row_df = pd.DataFrame([new_row])

    # Concatenate the new row to the existing DataFrame
    df = pd.concat([df, new_row_df], ignore_index=True)
    return df

def main():
    file_path = input("Enter the path to the CSV file: ")
    df = read_csv(file_path)

    while True:
        df = add_new_row(df)
        cont = input("Do you want to add another row? (yes/no): ").strip().lower()
        if cont != 'yes':
            break

    df.to_csv(file_path, index=False)
    print("Updated CSV file successfully.")

if __name__ == "__main__":
    main()
