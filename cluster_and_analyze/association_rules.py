# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mlxtend.frequent_patterns import apriori, association_rules
# import os
# from data_loader import load_data

# def prepare_data_for_association_rules(data):
#     # Drop identifier and irrelevant columns
#     cols_to_drop = [
#         "RESP_FINAL", "VIDEOS", "filename", "Image_SkinCluster", "single_cluster"
#     ]
#     data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])

#     """
#     Prepare data for association rule mining by discretizing continuous variables
    
#     Parameters:
#     -----------
#     data : pd.DataFrame
#     df : pd.DataFrame
#         Input dataframe
        
#     Returns:
#     --------
#     pd.DataFrame
#         Prepared dataframe
#     """
#     if df is None:
#         return None
    
#     # Make a copy to avoid modifying the original
#     data = df.copy()
    
#     # Identify categorical and numerical columns
#     categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and col != 'RESP_FINAL' and col != 'filename']
#     numerical_cols = [col for col in data.columns if data[col].dtype in [np.float64, np.int64] and col != 'RESP_FINAL']
    
#     # Discretize numerical columns
#     for col in numerical_cols:
#         # Get number of unique values
#         n_unique = data[col].nunique()
        
#         # Skip if column has too few unique values (already discrete)
#         if n_unique <= 2:
#             continue
        
#         # Determine number of bins based on unique values
#         if n_unique < 4:
#             n_bins = n_unique
#             labels = [f'{col}_Q{i+1}' for i in range(n_bins)]
#         else:
#             n_bins = 4
#             labels = [f'{col}_Q1', f'{col}_Q2', f'{col}_Q3', f'{col}_Q4']
        
#         try:
#             # Create bins
#             data[f'{col}_bin'] = pd.qcut(data[col], q=n_bins, labels=labels, duplicates='drop')
#             # Drop original column
#             data.drop(col, axis=1, inplace=True)
#         except ValueError as e:
#             # If binning fails, skip this column
#             print(f"Warning: Could not create bins for column {col}: {str(e)}")
#             continue
    
#     return data

# def create_one_hot_encoded_data(data):
#     """
#     Create one-hot encoded data for association rule mining
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input dataframe
        
#     Returns:
#     --------
#     pd.DataFrame
#         One-hot encoded dataframe
#     """
#     if df is None:
#         return None
    
#     # Make a copy to avoid modifying the original
#     data = df.copy()
    
#     # Remove identifier columns
#     if 'RESP_FINAL' in data.columns:
#         data.drop('RESP_FINAL', axis=1, inplace=True)
    
#     if 'filename' in data.columns:
#         data.drop('filename', axis=1, inplace=True)
    
#     # Get dummies for all columns
#     encoded_data = pd.get_dummies(data)
    
#     return encoded_data

# def mine_association_rules(df, min_support=0.1, min_confidence=0.5, min_lift=1.0):
#     """
#     Mine association rules from the data
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input dataframe
#     min_support : float
#         Minimum support threshold
#     min_confidence : float
#         Minimum confidence threshold
#     min_lift : float
#         Minimum lift threshold
        
#     Returns:
#     --------
#     pd.DataFrame
#         Association rules
#     """
#     if df is None:
#         return None
    
#     # Prepare data
#     prepared_data = prepare_data_for_association_rules(df)
#     if prepared_data is None:
#         print("Failed to prepare data")
#         return None
    
#     # Create one-hot encoded data
#     encoded_data = create_one_hot_encoded_data(prepared_data)
#     if encoded_data is None:
#         print("Failed to create one-hot encoded data")
#         return None
    
#     print(f"Mining association rules with min_support={min_support}, min_confidence={min_confidence}, min_lift={min_lift}...")
    
#     # Find frequent itemsets
#     frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)
    
#     # If no frequent itemsets found, try with lower support
#     if len(frequent_itemsets) == 0:
#         print(f"No frequent itemsets found with min_support={min_support}. Trying with lower support...")
#         min_support = min_support / 2
#         frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)
    
#     print(f"Found {len(frequent_itemsets)} frequent itemsets")
    
#     # If still no frequent itemsets found, return None
#     if len(frequent_itemsets) == 0:
#         print("No frequent itemsets found")
#         return None
    
#     # Generate association rules
#     rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    
#     # Filter by lift
#     rules = rules[rules['lift'] >= min_lift]
    
#     print(f"Found {len(rules)} association rules")
    
#     # If no rules found, try with lower confidence
#     if len(rules) == 0:
#         print(f"No rules found with min_confidence={min_confidence}. Trying with lower confidence...")
#         min_confidence = min_confidence / 2
#         rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
#         rules = rules[rules['lift'] >= min_lift]
#         print(f"Found {len(rules)} association rules with lower confidence")
    
#     return rules

# def analyze_rules(rules, top_n=20):
#     """
#     Analyze association rules
    
#     Parameters:
#     -----------
#     rules : pd.DataFrame
#         Association rules
#     top_n : int
#         Number of top rules to analyze
        
#     Returns:
#     --------
#     pd.DataFrame
#         Top rules
#     """
#     if rules is None or len(rules) == 0:
#         return None
    
#     # Sort rules by lift
#     sorted_rules = rules.sort_values('lift', ascending=False)
    
#     # Get top N rules
#     top_rules = sorted_rules.head(top_n)
    
#     print(f"\nTop {len(top_rules)} rules by lift:")
#     for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
#         antecedents = ', '.join(list(rule['antecedents']))
#         consequents = ', '.join(list(rule['consequents']))
#         print(f"{i}. {antecedents} -> {consequents} (support: {rule['support']:.4f}, confidence: {rule['confidence']:.4f}, lift: {rule['lift']:.4f})")
    
#     # Plot rule metrics
#     plt.figure(figsize=(10, 6))
#     plt.scatter(rules['support'], rules['confidence'], alpha=0.5, s=rules['lift']*20)
#     plt.xlabel('Support')
#     plt.ylabel('Confidence')
#     plt.title('Support vs Confidence (size represents lift)')
#     plt.tight_layout()
    
#     # Create plots directory if it doesn't exist
#     if not os.path.exists('plots'):
#         os.makedirs('plots')
    
#     plt.savefig('plots/rule_metrics.png')
#     plt.close()
    
#     # Save rules to CSV
#     if not os.path.exists('results'):
#         os.makedirs('results')
    
#     # Convert frozensets to strings for CSV export
#     rules_for_export = rules.copy()
#     rules_for_export['antecedents'] = rules_for_export['antecedents'].apply(lambda x: ', '.join(list(x)))
#     rules_for_export['consequents'] = rules_for_export['consequents'].apply(lambda x: ', '.join(list(x)))
    
#     rules_for_export.to_csv('results/association_rules.csv', index=False)
    
#     print(f"\nAll {len(rules)} rules saved to 'results/association_rules.csv'")
    
#     return top_rules

# def run_association_rule_mining(df=None, min_support=0.1, min_confidence=0.5, min_lift=1.0):
#     """
#     Run association rule mining on the data
    
#     Parameters:
#     -----------
#     df : pd.DataFrame, optional
#         Input dataframe. If None, data will be loaded from file.
#     min_support : float
#         Minimum support threshold
#     min_confidence : float
#         Minimum confidence threshold
#     min_lift : float
#         Minimum lift threshold
        
#     Returns:
#     --------
#     pd.DataFrame
#         Association rules
#     """
#     # Load data if not provided
#     if df is None:
#         df = load_data()
#         if df is None:
#             print("Failed to load data")
#             return None
    
#     # Mine association rules
#     rules = mine_association_rules(df, min_support, min_confidence, min_lift)
#     if rules is None or len(rules) == 0:
#         print("No association rules found")
#         return None
    
#     # Analyze rules
#     top_rules = analyze_rules(rules)
    
#     print("\nAssociation rule mining complete.")
    
#     return rules

# if __name__ == "__main__":
#     # Run association rule mining
#     run_association_rule_mining()
