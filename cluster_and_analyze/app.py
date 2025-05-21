import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np

# Import your modules - use absolute imports to avoid path issues
import sys
import os.path

# Add the current directory to the path to ensure modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import data_loader
import exploratory_analysis
import clustering
import dimensionality_reduction
import association_rules

st.set_page_config(page_title="MCB Data Analysis Dashboard", layout="wide")

st.title("MCB Hair & Skin Data Analysis Dashboard")

# Sidebar navigation
menu = st.sidebar.radio("Go to", [
    "Data Overview",
    "Exploratory Data Analysis",
    "Clustering",
    "Dimensionality Reduction",
    "Association Rules"
])

# Data loading
@st.cache_data
def load_data():
    # Look for the data file in ../data/MCB_DATA_MERGED.csv relative to this script
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/MCB_DATA_MERGED.csv"))
    if os.path.exists(data_path):
        return data_loader.load_data(data_path)
    else:
        return None

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Upload your data CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

if df is None:
    st.error("Data could not be loaded. Please upload a CSV file or check that '../data/MCB_DATA_MERGED.csv' exists.")
    st.stop()

if menu == "Data Overview":
    st.header("Data Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Missing values:")
    st.write(df.isnull().sum())

elif menu == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    eda_options = st.multiselect(
        "Select analyses to display",
        ["Age Analysis", "Ethnicity Analysis", "Correlation Matrix", "Category Distribution"]
    )
    
    if "Age Analysis" in eda_options:
        st.subheader("Age Analysis")
        figs = exploratory_analysis.plot_age_analysis(df)
        if figs:
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True)
                
    if "Ethnicity Analysis" in eda_options:
        st.subheader("Ethnicity Analysis")
        figs = exploratory_analysis.plot_ethnicity_analysis(df)
        if figs:
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True)
                
    if "Correlation Matrix" in eda_options:
        st.subheader("Correlation Matrix")
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        fig = exploratory_analysis.plot_correlation_matrix(df, numerical_cols)
        st.plotly_chart(fig, use_container_width=True)
        
    if "Category Distribution" in eda_options:
        st.subheader("Category Analysis")
        figs = exploratory_analysis.plot_category_analysis(df)
        if figs:
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True)

elif menu == "Clustering":
    st.header("Clustering")
    
    # Sidebar options
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    method = st.selectbox("Clustering Method", ["KMeans", "Agglomerative"])
    view_mode = st.radio("Visualization", ["2D", "3D"], horizontal=True)
    
    # Initialize session state for feature selection if not exists
    if 'cluster_x_index' not in st.session_state:
        st.session_state.cluster_x_index = 0
    if 'cluster_y_index' not in st.session_state:
        st.session_state.cluster_y_index = 1
    if 'cluster_z_index' not in st.session_state:
        st.session_state.cluster_z_index = 2
    
    # Select only numerical columns and exclude identifier/irrelevant columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Columns to exclude (identifiers and irrelevant columns)
    exclude_cols = ['RESP_FINAL', 'VIDEOS', 'filename', 'single_cluster', 'Image_SkinCluster', 'SCF1R']
    for col in exclude_cols:
        if col in numerical_cols:
            numerical_cols.remove(col)
    # Show which features are being used for clustering
    st.info(f"Using {len(numerical_cols)} numerical features for clustering: {', '.join(numerical_cols)}")
    X = df[numerical_cols].dropna()
    
    # Find optimal clusters first
    if st.button("Find Optimal Number of Clusters"):
        with st.spinner("Finding optimal clusters..."):
            try:
                # Try to use the function from the clustering module
                optimal_result = clustering.determine_optimal_clusters(X.values)
                if optimal_result and 'fig' in optimal_result:
                    st.plotly_chart(optimal_result['fig'], use_container_width=True)
                    st.info(f"Recommended clusters: {optimal_result['silhouette_k']} (silhouette) or {optimal_result['elbow_k']} (elbow)")
            except AttributeError:
                # If the function is not available, implement it directly here
                #st.warning("Using built-in optimal cluster detection...")
                
                # Initialize results
                results = {'k': [], 'inertia': [], 'silhouette': []}
                max_clusters = 10
                
                # Try different numbers of clusters
                for k in range(2, min(max_clusters + 1, len(X.values))):
                    # Fit KMeans
                    from sklearn.cluster import KMeans
                    from sklearn.metrics import silhouette_score
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(X.values)
                    
                    # Get inertia
                    inertia = kmeans.inertia_
                    
                    # Get silhouette score
                    if len(np.unique(kmeans.labels_)) > 1:  # Silhouette requires at least 2 clusters
                        silhouette = silhouette_score(X.values, kmeans.labels_)
                    else:
                        silhouette = 0
                    
                    # Store results
                    results['k'].append(k)
                    results['inertia'].append(inertia)
                    results['silhouette'].append(silhouette)
                
                # Find optimal k
                # For silhouette method, find the maximum score
                silhouette_k = results['k'][np.argmax(results['silhouette'])]
                
                # Create a side-by-side plot with subplots
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=['Elbow Method', 'Silhouette Method'])
                
                # Add elbow curve
                fig.add_trace(
                    go.Scatter(
                        x=results['k'],
                        y=results['inertia'],
                        mode='lines+markers',
                        name='Inertia',
                        marker=dict(color='blue', size=10),
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # Add silhouette curve
                fig.add_trace(
                    go.Scatter(
                        x=results['k'],
                        y=results['silhouette'],
                        mode='lines+markers',
                        name='Silhouette Score',
                        marker=dict(color='red', size=10),
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=2
                )
                
                # Update layout
                fig.update_layout(
                    title='Determining Optimal Number of Clusters',
                    height=500,
                    width=900,
                    showlegend=False,
                )
                
                # Update x and y axis labels
                fig.update_xaxes(title_text='Number of clusters (k)', row=1, col=1)
                fig.update_yaxes(title_text='Inertia', row=1, col=1)
                fig.update_xaxes(title_text='Number of clusters (k)', row=1, col=2)
                fig.update_yaxes(title_text='Silhouette Score', row=1, col=2)
                
                # Find optimal k for elbow method
                inertia_diff = np.diff(results['inertia'])
                inertia_diff2 = np.diff(inertia_diff)
                elbow_k = results['k'][np.argmax(inertia_diff2) + 1] if len(results['k']) > 2 else 2
                
                # Add vertical lines for optimal k values
                fig.add_vline(x=elbow_k, line_dash="dash", line_color="blue", row=1, col=1,
                             annotation_text=f"Optimal k={elbow_k}", annotation_position="top right")
                fig.add_vline(x=silhouette_k, line_dash="dash", line_color="red", row=1, col=2,
                             annotation_text=f"Optimal k={silhouette_k}", annotation_position="top right")
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"Recommended clusters: {silhouette_k} (based on silhouette score)")
                
                # Set optimal_result for later use
                optimal_result = {'silhouette_k': silhouette_k, 'elbow_k': silhouette_k}
    
    # Store data in session state for persistence
    if 'clustering_data' not in st.session_state:
        st.session_state.clustering_data = None
        st.session_state.clustering_labels = None
        st.session_state.clustering_features = None
    
    # Run clustering button
    if st.button("Run Clustering"):
        with st.spinner("Running clustering..."):
            # Run clustering
            if method == "KMeans":
                try:
                    # Try to use the function from the clustering module
                    labels = clustering.run_kmeans(X.values, n_clusters)
                except AttributeError:
                    # If the function is not available, implement it directly here
                    #st.warning("Using built-in KMeans implementation...")
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(X.values)
            else:
                try:
                    # Assuming you have a run_agglomerative function
                    labels = clustering.perform_hierarchical_clustering(X.values, n_clusters)['labels']
                except AttributeError:
                    # If the function is not available, implement it directly here
                    #st.warning("Using built-in Agglomerative Clustering implementation...")
                    from sklearn.cluster import AgglomerativeClustering
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = model.fit_predict(X.values)
            
            # Store results in session state
            st.session_state.clustering_data = X.values.copy()
            st.session_state.clustering_labels = labels
            st.session_state.clustering_features = X.columns.tolist() if hasattr(X, 'columns') else None
    
    # Only show visualizations if clustering has been run
    if st.session_state.clustering_data is not None and st.session_state.clustering_labels is not None:
        # Get data from session state
        data = st.session_state.clustering_data
        labels = st.session_state.clustering_labels
        # Ensure labels are integers for proper coloring
        if labels is not None:
            labels = labels.astype(int)
        feature_list = st.session_state.clustering_features
        
        # Display cluster distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        fig_dist = px.pie(
            values=cluster_counts.values, 
            names=cluster_counts.index,
            title="Cluster Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={'names': 'Cluster', 'values': 'Count'}
        )
        fig_dist.update_traces(textposition='inside', textinfo='percent+label')
        fig_dist.update_layout(
            legend_title_text="Cluster",
            font=dict(size=12),
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Create columns for visualization and controls
        col1, col2 = st.columns([3, 1])
        
        # Feature selection in the sidebar
        with col2:
            st.markdown("### Select Features to Visualize")
            if view_mode == "2D":
                x_feature = st.selectbox("X-axis", feature_list, index=0)
                y_feature = st.selectbox("Y-axis", feature_list, index=1 if len(feature_list) > 1 else 0)
                
                # Get indices
                x_index = feature_list.index(x_feature)
                y_index = feature_list.index(y_feature)
                
                # Add a button to update the plot
                update_plot = st.button("Update Plot", key="update_cluster_viz")
            else:  # 3D
                x_feature = st.selectbox("X-axis", feature_list, index=0)
                y_feature = st.selectbox("Y-axis", feature_list, index=1 if len(feature_list) > 1 else 0)
                z_feature = st.selectbox("Z-axis", feature_list, index=2 if len(feature_list) > 2 else 0)
                
                # Get indices
                x_index = feature_list.index(x_feature)
                y_index = feature_list.index(y_feature)
                z_index = feature_list.index(z_feature)
                
                # Add a button to update the plot
                update_plot = st.button("Update Plot", key="update_cluster_viz_3d")
        
        # Visualization
        with col1:
            if view_mode == "2D":
                try:
                    # Try to use the function from the clustering module
                    fig_2d = clustering.visualize_clusters_2d(
                        data, labels, feature_names=feature_list,
                        title=f"{method} Clustering (2D)",
                        x_index=x_index, y_index=y_index
                    )
                except AttributeError:
                    # If the function is not available, implement it directly here
                    #st.warning("Using built-in 2D cluster visualization...")
                    
                    # Create a fallback 2D visualization function
                    import plotly.express as px
                    import pandas as pd
                    
                    # Ensure indices are within bounds
                    x_index = min(x_index, data.shape[1]-1)
                    y_index = min(y_index, data.shape[1]-1)
                    
                    # Handle feature names
                    all_feature_names = ['Feature ' + str(i+1) for i in range(data.shape[1])]
                    if feature_list is not None:
                        if len(feature_list) >= 1:
                            all_feature_names = [str(name) for name in feature_list]
                    
                    # Get the selected feature names
                    x_name = all_feature_names[x_index]
                    y_name = all_feature_names[y_index]
                    
                    # Create dataframe for plotting with all features
                    df_plot = pd.DataFrame(data, columns=all_feature_names)
                    df_plot['Cluster'] = labels.astype(str)  # Convert to string for discrete colors
                    
                    # Create scatter plot with discrete colors
                    fig_2d = px.scatter(
                        df_plot, 
                        x=x_name, 
                        y=y_name,
                        color='Cluster',
                        title=f"{method} Clustering (2D)",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        hover_data={'Cluster': True}
                    )
                    
                    fig_2d.update_layout(
                        height=600,
                        width=800,
                        xaxis_title=x_name,
                        yaxis_title=y_name,
                        legend_title_text="Cluster"
                    )
                
                st.plotly_chart(fig_2d, use_container_width=True)
            else:
                try:
                    # Try to use the function from the clustering module
                    fig_3d = clustering.visualize_clusters_3d(
                        data, labels, feature_names=feature_list,
                        title=f"{method} Clustering (3D)",
                        x_index=x_index, y_index=y_index, z_index=z_index
                    )
                except AttributeError:
                    # If the function is not available, implement it directly here
                    #st.warning("Using built-in 3D cluster visualization...")
                    
                    # Create a fallback 3D visualization function
                    import plotly.express as px
                    import pandas as pd
                    
                    # Ensure indices are within bounds
                    x_index = min(x_index, data.shape[1]-1)
                    y_index = min(y_index, data.shape[1]-1)
                    z_index = min(z_index, data.shape[1]-1)
                    
                    # Handle feature names
                    all_feature_names = ['Feature ' + str(i+1) for i in range(data.shape[1])]
                    if feature_list is not None:
                        if len(feature_list) >= 1:
                            all_feature_names = [str(name) for name in feature_list]
                    
                    # Get the selected feature names
                    x_name = all_feature_names[x_index]
                    y_name = all_feature_names[y_index]
                    z_name = all_feature_names[z_index]
                    
                    # Create dataframe for plotting with all features
                    df_plot = pd.DataFrame(data, columns=all_feature_names)
                    df_plot['Cluster'] = labels.astype(str)  # Convert to string for discrete colors
                    
                    # Create 3D scatter plot with discrete colors
                    fig_3d = px.scatter_3d(
                        df_plot, 
                        x=x_name, 
                        y=y_name,
                        z=z_name,
                        color='Cluster',
                        title=f"{method} Clustering (3D)",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        hover_data={'Cluster': True}
                    )
                    
                    fig_3d.update_layout(
                        height=700,
                        width=800,
                        scene=dict(
                            xaxis_title=x_name,
                            yaxis_title=y_name,
                            zaxis_title=z_name
                        ),
                        legend_title_text="Cluster"
                    )
                
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # Add cluster labels to dataframe
            df_with_clusters = df.copy()
            df_with_clusters.loc[X.index, 'Cluster'] = labels
            
            # Show cluster characteristics
            st.subheader("Cluster Characteristics")
            for cluster in sorted(df_with_clusters['Cluster'].unique()):
                with st.expander(f"Cluster {cluster}"):
                    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
                    st.write(f"Size: {len(cluster_data)} samples ({len(cluster_data)/len(df_with_clusters)*100:.1f}%)")
                    st.write("Average values:")
                    st.dataframe(cluster_data[numerical_cols].mean().to_frame().T)

elif menu == "Dimensionality Reduction":
    st.header("Dimensionality Reduction")
    
    # Sidebar options
    dr_method = st.selectbox("Method", ["PCA", "t-SNE", "UMAP"])
    n_components = st.slider("Components", 2, 5, 2)
    use_clusters = st.checkbox("Color by clusters", value=True)
    n_clusters = st.slider("Number of clusters for coloring", 2, 10, 3) if use_clusters else None
    view_mode = st.radio("Visualization", ["2D", "3D"] if n_components >= 3 else ["2D"], horizontal=True)
    
    # Initialize session state for dimensionality reduction
    if 'dr_result' not in st.session_state:
        st.session_state.dr_result = None
    if 'dr_labels' not in st.session_state:
        st.session_state.dr_labels = None
    
    # Select only numerical columns and exclude identifier/irrelevant columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Columns to exclude (identifiers and irrelevant columns)
    exclude_cols = ['RESP_FINAL', 'VIDEOS', 'filename', 'single_cluster', 'Image_SkinCluster', 'SCF1R']
    for col in exclude_cols:
        if col in numerical_cols:
            numerical_cols.remove(col)
    # Show which features are being used for dimensionality reduction
    st.info(f"Using {len(numerical_cols)} features for dimensionality reduction: {', '.join(numerical_cols)}")
    X = df[numerical_cols].dropna()
    
    if st.button("Run Dimensionality Reduction"):
        with st.spinner(f"Running {dr_method}..."):
            # Run dimensionality reduction
            if dr_method == "PCA":
                result = dimensionality_reduction.perform_pca(X.values, n_components)
                # Show explained variance
                if 'fig' in result:
                    st.plotly_chart(result['fig'], use_container_width=True)
                    st.info(f"Total explained variance: {result['cumulative_variance'][-1]:.2%}")
            elif dr_method == "t-SNE":
                result = dimensionality_reduction.perform_tsne(X.values, n_components)
            else:  # UMAP
                result = dimensionality_reduction.perform_umap_reduction(X.values, n_components)
            
            # Generate cluster labels if requested
            labels = None
            if use_clusters:
                try:
                    # Try to use the function from the clustering module
                    labels = clustering.run_kmeans(X.values, n_clusters)
                except AttributeError:
                    # If the function is not available, implement it directly here
                    #st.warning("Using built-in KMeans implementation for dimensionality reduction...")
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(X.values)
                
                # Ensure labels are integers for proper coloring
                labels = labels.astype(int)
            
            # Define feature list for dimensionality reduction
            feature_list = X.columns.tolist() if hasattr(X, 'columns') else None
            
            # Store results in session state
            st.session_state.dr_result = result
            st.session_state.dr_labels = labels
            st.session_state.dr_features = feature_list
            
            # Visualize the reduced data
            if view_mode == "2D" or n_components < 3:
                # Use first 2 components for 2D visualization
                X_2d = result['transformed'][:, :2] if result['transformed'].shape[1] >= 2 else None
                
                if X_2d is not None:
                    fig_2d = dimensionality_reduction.plot_2d_reduction(
                        X_2d, labels=labels, method=dr_method, feature_names=feature_list
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)
            else:  # 3D view
                # Use first 3 components for 3D visualization
                X_3d = result['transformed'][:, :3] if result['transformed'].shape[1] >= 3 else None
                
                if X_3d is not None:
                    fig_3d = dimensionality_reduction.plot_3d_reduction(
                        X_3d, labels=labels, method=dr_method
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
            
            # Add reduced dimensions to dataframe
            reduced_df = pd.DataFrame(
                result['transformed'], 
                columns=[f"{dr_method}{i+1}" for i in range(result['transformed'].shape[1])],
                index=X.index
            )
            
            if use_clusters:
                reduced_df['Cluster'] = labels
            
            # Show the reduced data
            with st.expander("View reduced data"):
                st.dataframe(reduced_df)

elif menu == "Association Rules":
    st.header("Association Rule Mining")
    min_support = st.slider("Min Support", 0.01, 0.5, 0.1)
    min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5)
    if st.button("Run Association Rule Mining"):
        rules = association_rules.mine_association_rules(df, min_support, min_confidence)
        st.write(rules)

st.sidebar.markdown("---")
st.sidebar.info("Developed for MCB Hair & Skin Data Analysis")
