import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path='/explore_brows/data/MCB_DATA_MERGED.csv'):
    """
    Load the MCB merged data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the merged CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    try:
        # Try with semicolon separator first (based on previous processing)
        df = pd.read_csv(file_path, sep=';')
        print(f"Successfully loaded data from {file_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try with comma separator if semicolon fails
        try:
            df = pd.read_csv(file_path, sep=',')
            print(f"Successfully loaded data with comma separator from {file_path}")
            return df
        except Exception as e2:
            print(f"Error with comma separator: {e2}")
            return None

def preprocess_data(df, encode_categorical=True, scale_numerical=True):
    """
    Preprocess the MCB data for analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    encode_categorical : bool
        Whether to encode categorical variables
    scale_numerical : bool
        Whether to scale numerical variables
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    dict
        Preprocessing metadata (encoders, scalers, etc.)
    """
    if df is None:
        return None, None
    
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Handle missing values
    print("Handling missing values...")
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64]:
            # Fill numerical missing values with median
            data[col] = data[col].fillna(data[col].median())
        else:
            # Fill categorical missing values with mode
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # Identify categorical and numerical columns
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and col != 'RESP_FINAL' and col != 'filename']
    numerical_cols = [col for col in data.columns if data[col].dtype in [np.float64, np.int64] and col != 'RESP_FINAL']
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Initialize preprocessing metadata
    metadata = {
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'encoders': {},
        'scaler': None
    }
    
    # Encode categorical variables
    if encode_categorical and categorical_cols:
        print("Encoding categorical variables...")
        for col in categorical_cols:
            le = LabelEncoder()
            data[f"{col}_encoded"] = le.fit_transform(data[col])
            metadata['encoders'][col] = le
    
    # Scale numerical variables
    if scale_numerical and numerical_cols:
        print("Scaling numerical variables...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numerical_cols])
        
        # Replace original columns with scaled values
        for i, col in enumerate(numerical_cols):
            data[f"{col}_scaled"] = scaled_data[:, i]
        
        metadata['scaler'] = scaler
    
    return data, metadata

def get_analysis_data(df, metadata):
    """
    Get data ready for analysis by selecting appropriate columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    metadata : dict
        Preprocessing metadata
        
    Returns:
    --------
    pd.DataFrame
        Data ready for analysis with encoded/scaled features
    """
    if df is None or metadata is None:
        return None
    
    # Select columns for analysis
    analysis_cols = ['RESP_FINAL']
    
    # Add encoded categorical columns
    for col in metadata['categorical_cols']:
        if f"{col}_encoded" in df.columns:
            analysis_cols.append(f"{col}_encoded")
    
    # Add scaled numerical columns
    for col in metadata['numerical_cols']:
        if f"{col}_scaled" in df.columns:
            analysis_cols.append(f"{col}_scaled")
        else:
            analysis_cols.append(col)
    
    return df[analysis_cols]

def get_feature_names(metadata):
    """
    Get feature names for analysis
    
    Parameters:
    -----------
    metadata : dict
        Preprocessing metadata
        
    Returns:
    --------
    list
        List of feature names
    """
    if metadata is None:
        return []
    
    feature_names = []
    
    # Add encoded categorical columns
    for col in metadata['categorical_cols']:
        feature_names.append(f"{col}_encoded")
    
    # Add scaled numerical columns
    for col in metadata['numerical_cols']:
        feature_names.append(f"{col}_scaled")
    
    return feature_names

if __name__ == "__main__":
    # Test the data loading and preprocessing
    df = load_data()
    if df is not None:
        processed_df, metadata = preprocess_data(df)
        analysis_df = get_analysis_data(processed_df, metadata)
        
        print("\nPreprocessed data sample:")
        print(processed_df.head())
        
        print("\nAnalysis data sample:")
        print(analysis_df.head())
        
        print("\nFeature names for analysis:")
        print(get_feature_names(metadata))
