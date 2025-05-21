import pandas as pd

def compare_resp_final(file1_path, file2_path):
    """
    Compare RESP_FINAL values between two CSV files.
    
    Parameters:
    file1_path (str): Path to the first CSV file
    file2_path (str): Path to the second CSV file
    
    Returns:
    dict: Dictionary containing common and unique RESP_FINAL values
    """
    print(f"Comparing RESP_FINAL values between:\n1. {file1_path}\n2. {file2_path}")
    
    try:
        # Read both CSV files
        df1 = pd.read_csv(file1_path, sep=';')
        df2 = pd.read_csv(file2_path, sep=';')
        
        # Extract RESP_FINAL values from both dataframes
        resp_final_1 = set(df1['RESP_FINAL'].astype(str))
        resp_final_2 = set(df2['RESP_FINAL'].astype(str))
        
        # Find common and unique values
        common_values = resp_final_1.intersection(resp_final_2)
        only_in_file1 = resp_final_1 - resp_final_2
        only_in_file2 = resp_final_2 - resp_final_1
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total RESP_FINAL values in {file1_path}: {len(resp_final_1)}")
        print(f"Total RESP_FINAL values in {file2_path}: {len(resp_final_2)}")
        print(f"Number of common RESP_FINAL values: {len(common_values)}")
        print(f"Number of RESP_FINAL values only in {file1_path}: {len(only_in_file1)}")
        print(f"Number of RESP_FINAL values only in {file2_path}: {len(only_in_file2)}")
        
        # Print common values (first 20 for brevity)
        if common_values:
            print(f"\nFirst 20 common RESP_FINAL values:")
            for i, value in enumerate(sorted(list(common_values))[:20]):
                print(f"{i+1}. {value}")
            
            if len(common_values) > 20:
                print(f"... and {len(common_values) - 20} more")
        
        # Print values only in file1 (first 20 for brevity)
        if only_in_file1:
            print(f"\nFirst 20 RESP_FINAL values only in {file1_path}:")
            for i, value in enumerate(sorted(list(only_in_file1))[:20]):
                print(f"{i+1}. {value}")
            
            if len(only_in_file1) > 20:
                print(f"... and {len(only_in_file1) - 20} more")
        
        # Print values only in file2 (first 20 for brevity)
        if only_in_file2:
            print(f"\nFirst 20 RESP_FINAL values only in {file2_path}:")
            for i, value in enumerate(sorted(list(only_in_file2))[:20]):
                print(f"{i+1}. {value}")
            
            if len(only_in_file2) > 20:
                print(f"... and {len(only_in_file2) - 20} more")
        
        # Save results to CSV files
        pd.DataFrame(sorted(list(common_values)), columns=['RESP_FINAL']).to_csv('common_resp_final.csv', index=False)
        pd.DataFrame(sorted(list(only_in_file1)), columns=['RESP_FINAL']).to_csv('only_in_file1_resp_final.csv', index=False)
        pd.DataFrame(sorted(list(only_in_file2)), columns=['RESP_FINAL']).to_csv('only_in_file2_resp_final.csv', index=False)
        
        print("\nResults saved to:")
        print("- common_resp_final.csv")
        print("- only_in_file1_resp_final.csv")
        print("- only_in_file2_resp_final.csv")
        
        # Return the results
        return {
            'common': common_values,
            'only_in_file1': only_in_file1,
            'only_in_file2': only_in_file2
        }
    
    except Exception as e:
        print(f"Error comparing files: {e}")
        return None

if __name__ == "__main__":
    # Default file paths
    file1_path = 'data/MCB_DATA_SKIN_FILTERED.csv'
    file2_path = 'data/MCB_DATA_FINAL_UNIQUE.csv'
    
    # Call the comparison function
    compare_resp_final(file1_path, file2_path)
