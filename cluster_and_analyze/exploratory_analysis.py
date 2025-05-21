import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import load_data, preprocess_data

def plot_distributions(df, categorical_cols, numerical_cols, figsize=(15, 10)):
    """
    Plot distributions of categorical and numerical variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical column names
    numerical_cols : list
        List of numerical column names
    figsize : tuple
        Figure size
    """
    # Plot categorical distributions
    if categorical_cols:
        plt.figure(figsize=figsize)
        for i, col in enumerate(categorical_cols, 1):
            plt.subplot(len(categorical_cols), 1, i)
            sns.countplot(data=df, x=col)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/categorical_distributions.png')
        plt.close()
    
    # Plot numerical distributions
    if numerical_cols:
        plt.figure(figsize=figsize)
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(len(numerical_cols), 1, i)
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig('plots/numerical_distributions.png')
        plt.close()

def plot_correlation_matrix(df, numerical_cols, figsize=(12, 8)):
    """
    Plot an interactive correlation matrix using Plotly
    Returns a Plotly figure
    """
    # Remove 'VIDEOS' and 'Image_SkinCluster' if present
    cols = [col for col in numerical_cols if col not in ['VIDEOS', 'Image_SkinCluster']]
    corr = df[cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create the heatmap
    fig = go.Figure()
    
    # Add heatmap trace
    fig.add_trace(
        go.Heatmap(
            z=corr.mask(mask),
            x=corr.columns,
            y=corr.columns,
            zmin=-1,
            zmax=1,
            colorscale='RdBu',
            text=corr.mask(mask).round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title='Correlation Matrix',
        width=800,
        height=800,
        xaxis_title='',
        yaxis_title='',
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'},
        template='plotly_white'
    )
    
    return fig

def plot_category_analysis(df):
    """
    Analyze and plot the distribution of categories and their relationships
    Returns a list of Plotly figures
    """
    figs = []
    
    # Category distribution
    category_counts = df['CATEGORY'].value_counts()
    fig1 = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        hole=0.3,
        textinfo='label+percent',
        hovertemplate='Category: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig1.update_layout(
        title='Hair Color Category Distribution',
        showlegend=True,
        height=500
    )
    figs.append(fig1)
    
    # Category vs Skin Cluster
    if 'eval_cluster' in df.columns:
        # Calculate percentages
        cat_skin = df.groupby(['CATEGORY', 'eval_cluster']).size().reset_index(name='count')
        total_by_category = df.groupby('CATEGORY').size()
        cat_skin['percentage'] = cat_skin.apply(
            lambda x: (x['count'] / total_by_category[x['CATEGORY']]) * 100, axis=1
        )
        
        fig2 = px.bar(cat_skin,
                      x='CATEGORY',
                      y='percentage',
                      color='eval_cluster',
                      title='Skin Cluster Distribution by Hair Category',
                      text=cat_skin['percentage'].round(1).astype(str) + '%',
                      hover_data=['count'],
                      labels={'eval_cluster': 'Skin Cluster'})
        
        fig2.update_layout(
            height=500,
            xaxis_title='Hair Category',
            yaxis_title='Percentage (%)',
            bargap=0.2,
            showlegend=True
        )
        
        figs.append(fig2)
    
    return figs

def plot_age_analysis(df, figsize=(15, 10), use_scf1r=True):
    """
    Analyze and plot how characteristics vary across age groups using Plotly
    Returns a list of Plotly figures
    """
    # Create age groups based on SCF1_MOY
    age_labels = {
        1: '24-39 years',
        2: '40-49 years',
        3: '50-59 years',
        4: '60+ years'
    }
    
    # Map the age groups
    df['AGE_GROUP'] = df['SCF1R'].map(age_labels)
    
    figs = []
    
    # Hair characteristics violin plots
    hair_cols = [col for col in df.columns if 'HAIR' in col]
    if hair_cols:
        for col in hair_cols:
            fig = px.violin(df, x='AGE_GROUP', y=col, box=True, 
                          title=f'{col} Distribution by Age Group',
                          color='AGE_GROUP')
            fig.update_layout(
                height=500,
                xaxis_title='Age Group',
                yaxis_title=col,
                showlegend=True
            )
            figs.append(fig)
    
    # Skin cluster distribution
    if 'eval_cluster' in df.columns:
        # Convert eval_cluster to string for better display
        df['Skin Cluster'] = 'Cluster ' + df['eval_cluster'].astype(str)
        
        fig = px.histogram(df, x='AGE_GROUP', color='Skin Cluster',
                          title='Skin Cluster Distribution by Age Group',
                          barmode='group',
                          text_auto=True)
        
        fig.update_layout(
            height=500,
            xaxis_title='Age Group',
            yaxis_title='Count',
            bargap=0.2,
            bargroupgap=0.1
        )
        
        # Add percentage annotations
        total_by_age = df.groupby('AGE_GROUP').size()
        for age_group in df['AGE_GROUP'].unique():
            age_data = df[df['AGE_GROUP'] == age_group]
            cluster_counts = age_data['Skin Cluster'].value_counts()
            total = total_by_age[age_group]
            for cluster in cluster_counts.index:
                pct = (cluster_counts[cluster] / total) * 100
                fig.add_annotation(
                    x=age_group,
                    y=cluster_counts[cluster],
                    text=f'{pct:.1f}%',
                    showarrow=False,
                    yshift=10
                )
        
        figs.append(fig)
    
    return figs
    """
    Analyze and plot how characteristics vary across age groups
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    figsize : tuple
        Figure size
    use_scf1r : bool
        Whether to use SCF1R column for age groups instead of binning SCF1_MOY
    """
    # Check if we should use SCF1R for age groups
    if use_scf1r and 'SCF1R' in df.columns:
        print("Using SCF1R for age groups")
        # Create age group mapping
        age_group_map = {
            1: '25-39',
            2: '40-49',
            3: '50-59',
            4: '60-69'
        }
        # Map SCF1R to age groups
        df['AGE_GROUP'] = df['SCF1R'].map(age_group_map)
    elif 'SCF1_MOY' in df.columns:
        # Create age groups from SCF1_MOY
        print("Using SCF1_MOY for age groups")
        df['AGE_GROUP'] = pd.cut(df['SCF1_MOY'], bins=[20, 30, 40, 50, 60, 70], 
                                 labels=['20-30', '30-40', '40-50', '50-60', '60-70'])
    else:
        print("Neither age column (SCF1R or SCF1_MOY) found in dataframe")
        return
    
    # Analyze hair characteristics across age groups
    hair_cols = [col for col in df.columns if 'HAIR' in col]
    
    figs = []
    if hair_cols:
        fig, axes = plt.subplots(len(hair_cols), 1, figsize=figsize)
        if len(hair_cols) == 1:
            axes = [axes]
        for i, col in enumerate(hair_cols):
            sns.boxplot(data=df, x='AGE_GROUP', y=col, ax=axes[i])
            axes[i].set_title(f'{col} by Age Group')
        plt.tight_layout()
        import os
        os.makedirs('plots', exist_ok=True)
        fig.savefig('plots/hair_by_age.png')
        figs.append(fig)
    # Plot eval_cluster by AGE_GROUP
    if 'eval_cluster' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='AGE_GROUP', hue='eval_cluster', ax=ax2)
        ax2.set_title('Skin Cluster by Age Group')
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        fig2.savefig('plots/skin_by_age.png')
        figs.append(fig2)
    if figs:
        return figs
    else:
        return None

    
    # Analyze skin characteristics across age groups
    if 'eval_cluster' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='AGE_GROUP', hue='eval_cluster')
        plt.title('Skin Cluster by Age Group')
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)

def plot_ethnicity_analysis(df, figsize=(15, 10)):
    """
    Analyze and plot how characteristics vary across ethnicities using Plotly
    Returns a list of Plotly figures
    """
    # Map ethnicity codes to names for better readability
    ethnicity_map = {
        1: 'Caucasian/White',
        2: 'Black/African American',
        3: 'Asian',
        4: 'Hispanic/Latino',
        5: 'Native American',
        7: 'Other'
    }
    
    # Create a copy of the dataframe with mapped ethnicities
    df_eth = df.copy()
    df_eth['ETHNICITY'] = df_eth['ETHNI_USR'].map(ethnicity_map)
    
    figs = []
    
    # Hair characteristics violin plots
    hair_cols = [col for col in df.columns if 'HAIR' in col]
    if hair_cols:
        for col in hair_cols:
            fig = px.violin(df_eth, x='ETHNICITY', y=col, box=True,
                          title=f'{col} Distribution by Ethnicity',
                          color='ETHNICITY')
            fig.update_layout(
                height=500,
                xaxis_title='Ethnicity',
                yaxis_title=col,
                showlegend=True,
                xaxis_tickangle=45
            )
            figs.append(fig)
    
    # Skin cluster distribution
    if 'eval_cluster' in df_eth.columns:
        # Calculate percentages for each ethnicity-cluster combination
        total_by_ethnicity = df_eth.groupby('ETHNICITY').size()
        pct_data = []
        
        for ethnicity in df_eth['ETHNICITY'].unique():
            eth_data = df_eth[df_eth['ETHNICITY'] == ethnicity]
            total = len(eth_data)
            for cluster in range(1, 7):  # Clusters 1-6
                count = len(eth_data[eth_data['eval_cluster'] == cluster])
                pct = (count / total * 100) if total > 0 else 0
                pct_data.append({
                    'ETHNICITY': ethnicity,
                    'Skin Cluster': f'Cluster {cluster}',
                    'Percentage': pct,
                    'Count': count
                })
        
        df_pct = pd.DataFrame(pct_data)
        
        # Create the percentage distribution plot
        fig = px.bar(df_pct,
                     x='ETHNICITY',
                     y='Percentage',
                     color='Skin Cluster',
                     title='Skin Cluster Distribution by Ethnicity',
                     text=df_pct['Percentage'].round(1).astype(str) + '%',
                     hover_data={'Count': True},
                     category_orders={
                         'Skin Cluster': [f'Cluster {i}' for i in range(1, 7)]
                     })
        
        fig.update_layout(
            height=600,
            xaxis_title='Ethnicity',
            yaxis_title='Percentage (%)',
            bargap=0.2,
            bargroupgap=0.1,
            xaxis_tickangle=45,
            showlegend=True,
            legend_title_text='Skin Cluster',
            # Add a note about hover for counts
            annotations=[dict(
                text='Hover over bars to see exact counts',
                xref='paper', yref='paper',
                x=1, y=-0.2,
                showarrow=False,
                font=dict(size=10)
            )]
        )
        
        # Improve text visibility
        fig.update_traces(
            textposition='auto',
            textfont=dict(size=10),
            hovertemplate='<b>%{x}</b><br>' +
                          'Cluster: %{customdata[0]}<br>' +
                          'Percentage: %{y:.1f}%<br>' +
                          'Count: %{customdata[1]}<extra></extra>'
        )
        
        figs.append(fig)
    
    return figs
    plt.title('Skin Cluster by Ethnicity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/skin_by_ethnicity.png')
    plt.close()

def plot_hair_skin_relationship(df, figsize=(12, 8)):
    """
    Analyze and plot relationships between hair and skin characteristics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    figsize : tuple
        Figure size
    """
    if 'eval_cluster' not in df.columns:
        print("Skin cluster column (eval_cluster) not found in dataframe")
        return
    
    hair_cols = [col for col in df.columns if 'HAIR' in col]
    
    if not hair_cols:
        print("No hair columns found in dataframe")
        return
    
    # Plot relationships
    plt.figure(figsize=figsize)
    for i, col in enumerate(hair_cols, 1):
        plt.subplot(len(hair_cols), 1, i)
        sns.boxplot(data=df, x='eval_cluster', y=col)
        plt.title(f'{col} by Skin Cluster')
    plt.tight_layout()
    plt.savefig('plots/hair_by_skin.png')
    plt.close()
    
    # Plot category by skin cluster
    if 'CATEGORY' in df.columns:
        plt.figure(figsize=(10, 6))
        cross_tab = pd.crosstab(df['CATEGORY'], df['eval_cluster'])
        cross_tab.plot(kind='bar', stacked=True, figsize=figsize)
        plt.title('Hair Color Category by Skin Cluster')
        plt.xlabel('Hair Color Category')
        plt.ylabel('Count')
        plt.legend(title='Skin Cluster')
        plt.tight_layout()
        plt.savefig('plots/category_by_skin.png')
        plt.close()

def run_exploratory_analysis(df=None, use_scf1r_for_age=True):
    """
    Run all exploratory analysis functions
    
    Parameters:
    -----------
    df : pd.DataFrame, optional
        Input dataframe. If None, data will be loaded from file.
    """
    import os
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Load data if not provided
    if df is None:
        df = load_data()
        if df is None:
            print("Failed to load data")
            return
    
    # Identify categorical and numerical columns
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'RESP_FINAL' and col != 'filename']
    numerical_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64] and col != 'RESP_FINAL']
    
    # Run analysis functions
    print("Plotting distributions...")
    plot_distributions(df, categorical_cols, numerical_cols)
    
    print("Plotting correlation matrix...")
    plot_correlation_matrix(df, numerical_cols)
    
    print("Analyzing age relationships...")
    plot_age_analysis(df, use_scf1r=use_scf1r_for_age)
    
    print("Analyzing ethnicity relationships...")
    plot_ethnicity_analysis(df)
    
    print("Analyzing hair-skin relationships...")
    plot_hair_skin_relationship(df)
    
    print("Exploratory analysis complete. Plots saved to 'plots' directory.")

if __name__ == "__main__":
    # Run exploratory analysis
    run_exploratory_analysis()
