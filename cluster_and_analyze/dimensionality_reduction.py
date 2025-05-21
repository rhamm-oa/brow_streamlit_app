import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
from data_loader import load_data, preprocess_data, get_analysis_data, get_feature_names

def perform_pca(X, n_components=2):
    """
    Perform PCA dimensionality reduction
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    n_components : int
        Number of components
        
    Returns:
    --------
    dict
        Dictionary with PCA results and Plotly figure
    """
    if X is None or len(X) == 0:
        return None
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Get explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"PCA with {n_components} components:")
    print(f"Explained variance ratio: {explained_variance}")
    print(f"Cumulative explained variance: {cumulative_variance[-1]:.4f}")
    
    # Create Plotly figure for explained variance
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for individual explained variance
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(explained_variance) + 1)),
            y=explained_variance,
            name="Individual",
            marker_color='rgb(55, 83, 109)',
            opacity=0.7,
            text=[f"{v:.1%}" for v in explained_variance],
            textposition="auto"
        ),
        secondary_y=False,
    )
    
    # Add line chart for cumulative explained variance
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_variance) + 1)),
            y=cumulative_variance,
            name="Cumulative",
            marker_color='rgb(26, 118, 255)',
            mode="lines+markers"
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title="Explained Variance by Principal Components",
        xaxis_title="Principal Components",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)"),
        bargap=0.15,
        height=500,
        width=800
    )
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Individual Explained Variance", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)
    
    return {
        'model': pca,
        'transformed': X_pca,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'fig': fig
    }

def perform_tsne(X, n_components=2, perplexity=30, random_state=42):
    """
    Perform t-SNE dimensionality reduction
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    n_components : int
        Number of components
    perplexity : float
        Perplexity parameter for t-SNE
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with t-SNE results
    """
    if X is None or len(X) == 0:
        return None
    
    # Perform t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    
    print(f"t-SNE with {n_components} components and perplexity {perplexity} complete.")
    
    return {
        'model': tsne,
        'transformed': X_tsne
    }

def perform_umap_reduction(X, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Perform UMAP dimensionality reduction
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    n_components : int
        Number of components
    n_neighbors : int
        Number of neighbors for UMAP
    min_dist : float
        Minimum distance parameter for UMAP
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with UMAP results
    """
    if X is None or len(X) == 0:
        return None
    
    # Perform UMAP
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                        min_dist=min_dist, random_state=random_state)
    X_umap = reducer.fit_transform(X)
    
    print(f"UMAP with {n_components} components, {n_neighbors} neighbors, and {min_dist} min_dist complete.")
    
    return {
        'model': reducer,
        'transformed': X_umap
    }

def plot_2d_reduction(X_reduced, labels=None, method='PCA', feature_names=None):
    """
    Plot 2D dimensionality reduction results using Plotly
    
    Parameters:
    -----------
    X_reduced : np.ndarray
        Reduced feature matrix (2D)
    labels : np.ndarray, optional
        Cluster labels
    method : str
        Dimensionality reduction method name
    feature_names : list, optional
        Original feature names for hover information
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    if X_reduced is None or X_reduced.shape[1] < 2:
        return None
    
    # Create dataframe for plotting
    df_plot = pd.DataFrame({
        f'{method}1': X_reduced[:, 0],
        f'{method}2': X_reduced[:, 1],
    })
    
    if labels is not None:
        df_plot['Cluster'] = labels.astype(str)  # Convert to string for discrete colors
        # Create scatter plot with cluster colors
        fig = px.scatter(
            df_plot, 
            x=f'{method}1', 
            y=f'{method}2',
            color='Cluster',
            title=f'2D {method} Projection',
            labels={f'{method}1': f'{method} Component 1', f'{method}2': f'{method} Component 2'},
            color_discrete_sequence=px.colors.qualitative.Bold,  # More vibrant color theme
            hover_name=df_plot.index if len(df_plot) < 1000 else None
        )
        fig.update_layout(legend_title_text="Cluster")
    else:
        # Create scatter plot without cluster colors
        fig = px.scatter(
            df_plot, 
            x=f'{method}1', 
            y=f'{method}2',
            title=f'2D {method} Projection',
            labels={f'{method}1': f'{method} Component 1', f'{method}2': f'{method} Component 2'},
            hover_name=df_plot.index if len(df_plot) < 1000 else None
        )
    
    # Add feature vectors if PCA and feature names are provided
    if method == 'PCA' and feature_names is not None:
        # Get the PCA loadings (feature vectors)
        # This is a placeholder - in real implementation you would pass the PCA model
        # and get the components_ attribute
        # For now, we'll just create random vectors for demonstration
        pass
    
    fig.update_layout(
        height=600,
        width=800,
        template='plotly_white',
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='gray'),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='gray')
    )
    
    return fig

def plot_3d_reduction(X_reduced, labels=None, method='PCA'):
    """
    Plot 3D dimensionality reduction results using Plotly
    
    Parameters:
    -----------
    X_reduced : np.ndarray
        Reduced feature matrix (3D)
    labels : np.ndarray, optional
        Cluster labels
    method : str
        Dimensionality reduction method name
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    if X_reduced is None or X_reduced.shape[1] != 3:
        print(f"Error: Expected 3D data, got {X_reduced.shape[1]}D")
        return None
    
    # Create dataframe for plotting
    df_plot = pd.DataFrame({
        f'{method}1': X_reduced[:, 0],
        f'{method}2': X_reduced[:, 1],
        f'{method}3': X_reduced[:, 2]
    })
    
    if labels is not None:
        df_plot['Cluster'] = labels.astype(str)  # Convert to string for discrete colors
        # Create 3D scatter plot with cluster colors
        fig = px.scatter_3d(
            df_plot, 
            x=f'{method}1', 
            y=f'{method}2', 
            z=f'{method}3',
            color='Cluster',
            title=f'3D {method} Projection',
            labels={
                f'{method}1': f'{method} Component 1', 
                f'{method}2': f'{method} Component 2',
                f'{method}3': f'{method} Component 3'
            },
            color_discrete_sequence=px.colors.qualitative.Bold,  # More vibrant color theme
            hover_name=df_plot.index if len(df_plot) < 1000 else None
        )
        fig.update_layout(legend_title_text="Cluster")
    else:
        # Create 3D scatter plot without cluster colors
        fig = px.scatter_3d(
            df_plot, 
            x=f'{method}1', 
            y=f'{method}2', 
            z=f'{method}3',
            title=f'3D {method} Projection',
            labels={
                f'{method}1': f'{method} Component 1', 
                f'{method}2': f'{method} Component 2',
                f'{method}3': f'{method} Component 3'
            },
            hover_name=df_plot.index if len(df_plot) < 1000 else None
        )
    
    fig.update_layout(
        height=700,
        width=900,
        scene=dict(
            xaxis=dict(showbackground=True),
            yaxis=dict(showbackground=True),
            zaxis=dict(showbackground=True)
        )
    )
    
    return fig

def run_dimensionality_reduction(df=None, labels=None):
    """
    Run dimensionality reduction on the data
    
    Parameters:
    -----------
    df : pd.DataFrame, optional
        Input dataframe. If None, data will be loaded from file.
    labels : np.ndarray, optional
        Cluster labels
        
    Returns:
    --------
    dict
        Dictionary with dimensionality reduction results
    """
    # Load data if not provided
    if df is None:
        df = load_data()
        if df is None:
            print("Failed to load data")
            return None
    
    # Preprocess data
    processed_df, metadata = preprocess_data(df)
    if processed_df is None:
        print("Failed to preprocess data")
        return None
    
    # Get data for analysis
    analysis_df = get_analysis_data(processed_df, metadata)
    if analysis_df is None:
        print("Failed to get analysis data")
        return None
    
    # Get feature matrix (exclude RESP_FINAL)
    X = analysis_df.drop('RESP_FINAL', axis=1).values
    
    # Load cluster labels if not provided
    if labels is None:
        try:
            clustered_df = pd.read_csv('results/clustered_data.csv')
            if 'KMeans_Cluster' in clustered_df.columns:
                labels = clustered_df['KMeans_Cluster'].values
                print("Loaded cluster labels from 'results/clustered_data.csv'")
        except:
            print("No cluster labels found. Proceeding without labels.")
    
    # Perform PCA
    print("\nPerforming PCA...")
    pca_results_2d = perform_pca(X, n_components=2)
    pca_results_3d = perform_pca(X, n_components=3)
    
    # Plot PCA results
    if pca_results_2d is not None:
        plot_2d_reduction(pca_results_2d['transformed'], labels, method='PCA')
    
    if pca_results_3d is not None:
        plot_3d_reduction(pca_results_3d['transformed'], labels, method='PCA')
    
    # Perform t-SNE
    print("\nPerforming t-SNE...")
    tsne_results_2d = perform_tsne(X, n_components=2)
    tsne_results_3d = perform_tsne(X, n_components=3)
    
    # Plot t-SNE results
    if tsne_results_2d is not None:
        plot_2d_reduction(tsne_results_2d['transformed'], labels, method='t-SNE')
    
    if tsne_results_3d is not None:
        plot_3d_reduction(tsne_results_3d['transformed'], labels, method='t-SNE')
    
    # Perform UMAP
    print("\nPerforming UMAP...")
    umap_results_2d = perform_umap_reduction(X, n_components=2)
    umap_results_3d = perform_umap_reduction(X, n_components=3)
    
    # Plot UMAP results
    if umap_results_2d is not None:
        plot_2d_reduction(umap_results_2d['transformed'], labels, method='UMAP')
    
    if umap_results_3d is not None:
        plot_3d_reduction(umap_results_3d['transformed'], labels, method='UMAP')
    
    print("\nDimensionality reduction complete.")
    
    return {
        'pca_2d': pca_results_2d,
        'pca_3d': pca_results_3d,
        'tsne_2d': tsne_results_2d,
        'tsne_3d': tsne_results_3d,
        'umap_2d': umap_results_2d,
        'umap_3d': umap_results_3d
    }

if __name__ == "__main__":
    # Run dimensionality reduction
    run_dimensionality_reduction()
