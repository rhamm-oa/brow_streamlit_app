import pandas as pd

# Read the CSV file
# Note: If the separator is not semicolon, you may need to adjust this
file_path = 'data/MCB_DATA_FINAL.csv'
print(f"Reading file: {file_path}")

try:
    # Try with semicolon separator first (based on your example)
    df = pd.read_csv(file_path, sep=';')
    
    # List of columns to keep
    columns_to_keep = [
        'RESP_FINAL', 'VIDEOS', 'CATEGORY', 'SCF1R', 'SCF1_MOY',
        'ETHNI_USR', 'HAIR_LENGTHR', 'HAIR_THICKNESSR', 'HAIR_TYPER', 'HAIR_GREYR'
    ]
    
    # Check if all required columns exist in the dataframe
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following columns are missing from the dataset: {missing_columns}")
        # Keep only columns that exist in the dataframe
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    
    # Filter for rows where VIDEOS=1 and keep only specified columns
    if 'VIDEOS' in df.columns:
        filtered_df = df[df['VIDEOS'] == 1][columns_to_keep]
    else:
        print("Warning: 'VIDEOS' column not found. Keeping all rows.")
        filtered_df = df[columns_to_keep]

    print("Unique RESP_FINAL values of filtered:",filtered_df['RESP_FINAL'].nunique())

    # Keep only the first occurrence of each RESP_FINAL value
    unique_df = filtered_df.drop_duplicates(subset=['RESP_FINAL'])
    
    # Save to a new CSV file
    output_file = 'data/MCB_DATA_UNIQUE.csv'
    unique_df.to_csv(output_file, index=False, sep=';')
    
    # Print information about the results
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Filtered data shape (VIDEOS=1): {filtered_df.shape}")
    print(f"Final unique data shape: {unique_df.shape}")
    
    # Show the first 10 rows
    print("\nFirst 10 rows of the unique data:")
    print(unique_df.head(10))
    
    # Verify uniqueness
    resp_final_counts = unique_df['RESP_FINAL'].value_counts()
    duplicates = resp_final_counts[resp_final_counts > 1]
    
    if len(duplicates) > 0:
        print("\nWarning: There are still duplicate RESP_FINAL values:")
        print(duplicates)
    else:
        print("\nSuccess: All RESP_FINAL values are unique!")
    
    print(f"\nOutput saved to: {output_file}")

except Exception as e:
    print(f"Error processing the file: {e}")
    
    # Try with comma separator if semicolon fails
    try:
        print("Trying with comma as separator...")
        df = pd.read_csv(file_path, sep=',')
        print("Successfully read with comma separator.")
        # Repeat the same processing...
        # (Code omitted for brevity - would be the same as above)
    except Exception as e2:
        print(f"Error with comma separator: {e2}")
        print("Please check the file format and try again.")
