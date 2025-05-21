import pandas as pd

def merge_datasets(file1_path, file2_path, output_path):
    """
    Merge two CSV files based on common RESP_FINAL values.
    
    Parameters:
    file1_path (str): Path to the first CSV file (MCB_DATA_FINAL_UNIQUE.csv)
    file2_path (str): Path to the second CSV file (MCB_DATA_SKIN_FILTERED.csv)
    output_path (str): Path to save the merged CSV file
    
    Returns:
    pandas.DataFrame: The merged dataframe
    """
    print(f"Merging datasets:\n1. {file1_path}\n2. {file2_path}")
    
    try:
        # Read both CSV files
        df1 = pd.read_csv(file1_path, sep=';')  # MCB_DATA_FINAL_UNIQUE
        df2 = pd.read_csv(file2_path, sep=';')  # MCB_DATA_SKIN_FILTERED
        
        print(f"\nDataset 1 shape: {df1.shape}")
        print(f"Dataset 1 columns: {df1.columns.tolist()}")
        print(f"\nDataset 2 shape: {df2.shape}")
        print(f"Dataset 2 columns: {df2.columns.tolist()}")
        
        # Perform an inner join on RESP_FINAL
        # This will keep only the rows where RESP_FINAL exists in both dataframes
        merged_df = pd.merge(df1, df2, on='RESP_FINAL', how='inner')
        
        print(f"\nMerged dataset shape: {merged_df.shape}")
        print(f"Merged dataset columns: {merged_df.columns.tolist()}")
        
        # Save the merged dataframe to a new CSV file
        merged_df.to_csv(output_path, sep=';', index=False)
        
        print(f"\nFirst 5 rows of the merged dataset:")
        print(merged_df.head(5))
        
        print(f"\nMerged dataset saved to: {output_path}")
        
        # Count of common RESP_FINAL values
        common_count = len(merged_df)
        print(f"\nNumber of common RESP_FINAL values: {common_count}")
        
        # Count of RESP_FINAL values only in file1
        only_in_file1_count = len(set(df1['RESP_FINAL']) - set(df2['RESP_FINAL']))
        print(f"Number of RESP_FINAL values only in {file1_path}: {only_in_file1_count}")
        
        # Count of RESP_FINAL values only in file2
        only_in_file2_count = len(set(df2['RESP_FINAL']) - set(df1['RESP_FINAL']))
        print(f"Number of RESP_FINAL values only in {file2_path}: {only_in_file2_count}")
        
        return merged_df
    
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return None

if __name__ == "__main__":
    # File paths
    file1_path = 'data/MCB_DATA_FINAL_UNIQUE.csv'  # First dataset with hair data
    file2_path = 'data/MCB_DATA_SKIN_FILTERED.csv'  # Second dataset with skin data
    output_path = 'data/MCB_DATA_MERGED.csv'  # Output file for the merged dataset
    
    # Merge the datasets
    merge_datasets(file1_path, file2_path, output_path)
