# MCB Data Analysis and Clustering

This project provides tools for analyzing and clustering hair and skin characteristics data from the MCB dataset.

## Project Structure

- `data_loader.py`: Functions to load and preprocess the MCB data
- `exploratory_analysis.py`: Exploratory data analysis and visualization
- `clustering.py`: Implementation of clustering algorithms (K-means, hierarchical)
- `dimensionality_reduction.py`: PCA and t-SNE for visualization
- `association_rules.py`: Mining association rules between features
- `app.py`: Streamlit dashboard for interactive visualization

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit dashboard:
   ```
   streamlit run app.py
   ```

## Features

- Data exploration and visualization
- Clustering analysis
- Dimensionality reduction
- Association rule mining
- Interactive dashboard

## Data Description

The dataset contains the following key columns:

- RESP_FINAL: Person identifier
- CATEGORY: Hair color category (Medium, Dark, Light)
- SCF1_MOY: Age (25-69)
- ETHNI_USR: Ethnicity (1-7, with 1 being Caucasian/White)
- HAIR_LENGTHR: Hair length (2-7, with 7 being long)
- HAIR_THICKNESSR: Hair thickness (1-5)
- HAIR_TYPER: Hair type (1-6)
- HAIR_GREYR: Gray hair level (1-6)
- eval_cluster: Skin tone cluster (1-6, with 6 being darkest)
