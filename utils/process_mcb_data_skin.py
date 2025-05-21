import pandas as pd

def process_mcb_data_skin():
    """
    Process the MCB_DATA_SKIN.csv file to extract only the specified columns:
    RESP_FINAL, filename, eval_cluster, single_cluster, Image_SkinCluster
    """
    file_path = 'data/MCB_DATA_SKIN.csv'
    print(f"Reading file: {file_path}")

    try:
        # Read the CSV file with semicolon separator
        df = pd.read_csv(file_path, sep=';')
        
        # List of columns to keep
        columns_to_keep = [
            'RESP_FINAL', 'filename', 'eval_cluster', 'single_cluster', 'Image_SkinCluster'
        ]
        
        # Check if all required columns exist in the dataframe
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        if missing_columns:
            print(f"Warning: The following columns are missing from the dataset: {missing_columns}")
            # Keep only columns that exist in the dataframe
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        
        # Keep only the specified columns
        filtered_df = df[columns_to_keep]
        
        # Save to a new CSV file
        output_file = 'data/MCB_DATA_SKIN_FILTERED.csv'
        filtered_df.to_csv(output_file, index=False, sep=';')
        
        # Print information about the results
        print(f"\nOriginal data shape: {df.shape}")
        print(f"Filtered data shape: {filtered_df.shape}")
        
        # Show the first 10 rows
        print("\nFirst 10 rows of the filtered data:")
        print(filtered_df.head(10))
        
        print(f"\nOutput saved to: {output_file}")
        
        return True
    except Exception as e:
        print(f"Error processing the file: {e}")
        
        # Try with comma separator if semicolon fails
        try:
            print("Trying with comma as separator...")
            df = pd.read_csv(file_path, sep=',')
            print("Successfully read with comma separator.")
            
            # Continue with the same processing as above
            # Keep only the specified columns
            filtered_df = df[columns_to_keep]
            
            # Save to a new CSV file
            output_file = 'MCB_DATA_SKIN_FILTERED.csv'
            filtered_df.to_csv(output_file, index=False, sep=';')
            
            # Print information about the results
            print(f"\nOriginal data shape: {df.shape}")
            print(f"Filtered data shape: {filtered_df.shape}")
            
            # Show the first 10 rows
            print("\nFirst 10 rows of the filtered data:")
            print(filtered_df.head(10))
            
            print(f"\nOutput saved to: {output_file}")
            
            return True
        except Exception as e2:
            print(f"Error with comma separator: {e2}")
            print("Please check the file format and try again.")
            return False

if __name__ == "__main__":
    process_mcb_data_skin()
