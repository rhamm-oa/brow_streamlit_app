import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

import os
from datetime import datetime
from plotly.subplots import make_subplots

output_dir = "/home/user/brow/analysis"
## Keep only the columns we are interested in for the all mcb hair data
full_df=pd.read_csv("/home/user/brow/data/MCB_data.csv",delimiter=";",encoding="utf-8")
columns_to_keep = [
    'RESP_FINAL', 
    'VIDEOS', 
    'CATEGORY', 
    'SCF1R',
    'SCF1_MOY',
    'ETHNI_USR',
    'HAIR_THICKNESSR',
    'HAIR_TYPER',
    'HAIR_CURL1R',
    'HAIR_CURL_2R',
    'HAIR_GREYR']
df1=full_df[columns_to_keep]
df2=pd.read_csv("/home/user/brow/data/mcbhair_skin_data_with_eval_clusters.csv",delimiter=";",encoding="utf-8")

################- Basic information -######################
print(f"df1 shape: {df1.shape}")
print(f"df2 shape: {df2.shape}")



###############- Get unique RESP_FINAL values ##############
df1_resp_unique = df1['RESP_FINAL'].unique()
df2_resp_unique = df2['RESP_FINAL'].unique()

df1_resp_set = set(df1_resp_unique)
df2_resp_set = set(df2_resp_unique)

# Find values in each set but not in the other
only_in_df1 = df1_resp_set - df2_resp_set
only_in_df2 = df2_resp_set - df1_resp_set
in_both = df1_resp_set.intersection(df2_resp_set)

print(f"Unique RESP_FINAL in df1: {len(df1_resp_set)}")
print(f"Unique RESP_FINAL in df2: {len(df2_resp_set)}")
print(f"RESP_FINAL only in df1: {len(only_in_df1)}")
print(f"RESP_FINAL only in df2: {len(only_in_df2)}")
print(f"RESP_FINAL in both: {len(in_both)}")

summary_data = pd.DataFrame({
    'Category': ['Only in df1', 'Only in df2', 'In both'],
    'Count': [len(only_in_df1), len(only_in_df2), len(in_both)]
})

# Save the missing RESP_FINAL values to CSV files
pd.DataFrame(list(only_in_df1), columns=['RESP_FINAL']).to_csv(f"{output_dir}/resp_final_only_in_df1.csv", index=False)
pd.DataFrame(list(only_in_df2), columns=['RESP_FINAL']).to_csv(f"{output_dir}/resp_final_only_in_df2.csv", index=False)

# 1. Matplotlib: Venn Diagram-like visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(summary_data['Category'], summary_data['Count'], color=['#ff9999', '#66b3ff', '#99ff99'])

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:,}', ha='center', va='bottom')

plt.title('Distribution of RESP_FINAL values across dataframes', fontsize=15)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/resp_final_distribution_bars.png", dpi=300)
plt.close()


# 2. Matplotlib: Histogram of RESP_FINAL value counts in df1
value_counts = df1['RESP_FINAL'].value_counts()

plt.figure(figsize=(12, 6))
plt.hist(value_counts, bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of RESP_FINAL occurrences in df1', fontsize=15)
plt.xlabel('Number of occurrences per RESP_FINAL', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/resp_final_occurrences_histogram.png", dpi=300)
plt.close()

# 3. Plotly: Interactive pie chart of RESP_FINAL distribution
fig = px.pie(summary_data, values='Count', names='Category', 
             title='Distribution of RESP_FINAL values',
             color_discrete_sequence=px.colors.qualitative.Set3)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.write_html(f"{output_dir}/resp_final_distribution_pie.html")

# 4. Plotly: Interactive bar chart of top RESP_FINAL occurrences
top_resp_final = value_counts.head(20)
fig = px.bar(x=top_resp_final.index, y=top_resp_final.values,
             title='Top 20 RESP_FINAL values by occurrence count',
             labels={'x': 'RESP_FINAL', 'y': 'Count'})
fig.update_layout(xaxis_type='category')
fig.write_html(f"{output_dir}/top_resp_final_occurrences.html")

# 5. Plotly: Histogram of occurrences with log scale
fig = px.histogram(x=value_counts.values, nbins=50,
                  title='Distribution of RESP_FINAL occurrences (log scale)',
                  labels={'x': 'Number of occurrences per RESP_FINAL', 'y': 'Frequency'})
fig.update_layout(yaxis_type="log")
fig.write_html(f"{output_dir}/resp_final_occurrences_histogram_log.html")

# 6. Matplotlib: Box plot of RESP_FINAL occurrences
plt.figure(figsize=(10, 6))
plt.boxplot(value_counts, vert=False, widths=0.7)
plt.title('Box plot of RESP_FINAL occurrences in df1', fontsize=15)
plt.xlabel('Number of occurrences', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/resp_final_occurrences_boxplot.png", dpi=300)
plt.close()


# 7. Plotly: Scatter plot comparing some metric across dataframes (if applicable)
# This assumes there's some numerical column in both dataframes to compare
# If not, this can be skipped
common_resp = list(in_both)
if len(common_resp) > 0 and 'some_numeric_column' in df1.columns and 'some_numeric_column' in df2.columns:
    # Get values for common RESP_FINAL
    df1_subset = df1[df1['RESP_FINAL'].isin(common_resp)].drop_duplicates(subset=['RESP_FINAL'])
    df2_subset = df2[df2['RESP_FINAL'].isin(common_resp)].drop_duplicates(subset=['RESP_FINAL'])
    
    # Merge to get paired values
    comparison_df = pd.merge(
        df1_subset[['RESP_FINAL', 'some_numeric_column']], 
        df2_subset[['RESP_FINAL', 'some_numeric_column']], 
        on='RESP_FINAL', 
        suffixes=('_df1', '_df2')
    )
    
    fig = px.scatter(comparison_df, x='some_numeric_column_df1', y='some_numeric_column_df2',
                    title='Comparison of values for common RESP_FINAL',
                    labels={'some_numeric_column_df1': 'Value in df1', 'some_numeric_column_df2': 'Value in df2'})
    fig.add_trace(go.Scatter(x=[0, max(comparison_df['some_numeric_column_df1'])], 
                            y=[0, max(comparison_df['some_numeric_column_df1'])],
                            mode='lines', line=dict(dash='dash', color='red'),
                            name='y=x line'))
    fig.write_html(f"{output_dir}/resp_final_value_comparison.html")

# 8. Create a summary report
with open(f"{output_dir}/analysis_summary.txt", 'w') as f:
    f.write(f"RESP_FINAL Analysis Summary\n")
    f.write(f"=========================\n\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Dataset 1 (df1):\n")
    f.write(f"  - Total rows: {df1.shape[0]}\n")
    f.write(f"  - Unique RESP_FINAL values: {len(df1_resp_set)}\n\n")
    f.write(f"Dataset 2 (df2):\n")
    f.write(f"  - Total rows: {df2.shape[0]}\n")
    f.write(f"  - Unique RESP_FINAL values: {len(df2_resp_set)}\n\n")
    f.write(f"Comparison:\n")
    f.write(f"  - RESP_FINAL values only in df1: {len(only_in_df1)}\n")
    f.write(f"  - RESP_FINAL values only in df2: {len(only_in_df2)}\n")
    f.write(f"  - RESP_FINAL values in both datasets: {len(in_both)}\n\n")
    
    if len(only_in_df2) > 0:
        f.write(f"Missing RESP_FINAL values (in df2 but not in df1):\n")
        for val in sorted(list(only_in_df2))[:20]:  # Show first 20
            f.write(f"  - {val}\n")
        if len(only_in_df2) > 20:
            f.write(f"  - ... and {len(only_in_df2) - 20} more\n")
        f.write(f"\nSee resp_final_only_in_df2.csv for complete list\n\n")
    
    # Occurrence statistics
    f.write(f"RESP_FINAL occurrence statistics in df1:\n")
    f.write(f"  - Min occurrences: {value_counts.min()}\n")
    f.write(f"  - Max occurrences: {value_counts.max()}\n")
    f.write(f"  - Mean occurrences: {value_counts.mean():.2f}\n")
    f.write(f"  - Median occurrences: {value_counts.median():.2f}\n\n")
    
    f.write(f"All visualizations and data files saved to: {output_dir}/\n")

print(f"\nAnalysis complete! All results saved to: {output_dir}/")
print(f"Open {output_dir}/analysis_summary.txt for a summary of findings.")