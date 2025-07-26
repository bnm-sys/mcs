import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from notebooks.data_utils import load_and_scale_data
import numpy as np
# Import the actual load_and_scale_data function from your notebooks.data_utils
try:
    from notebooks.data_utils import load_and_scale_data
except ImportError:
    st.error("Could not import 'load_and_scale_data' from 'notebooks/data_utils.py'. "
             "Please ensure the file exists in the 'notebooks' directory relative to your app.py, "
             "and that all necessary dependencies are installed.")
    st.stop()


REQUIRED_COLUMNS = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
CLUSTER_COL = 'Cluster'


def validate_columns(df):
    """Ensure required columns are present in the uploaded CSV."""
    return all(col in df.columns for col in REQUIRED_COLUMNS)


def scale_features(df, features):
    """Scale features using StandardScaler and return scaled numpy array.
    This function is used for uploaded files.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled


def perform_clustering(X_scaled, n_clusters, random_state=42):
    """Perform KMeans clustering and return cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                    max_iter=300, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    return labels


def plot_3d_clusters(df, features, cluster_col):
    """Create and return a Plotly 3D scatter plot figure."""
    fig = px.scatter_3d(
        df,
        x=features[0],
        y=features[1],
        z=features[2],
        color=cluster_col,
        title=f"3D Clustering Visualization (K={df[cluster_col].nunique()})",
        labels={features[0]: features[0], features[1]: features[1], features[2]: features[2]},
        width=800,
        height=600,
    )
    return fig


def get_cluster_profiles(df, cluster_col, features):
    """Return mean values of features per cluster."""
    return df.groupby(cluster_col)[features].mean()


def suggest_customer_type(profile, mean_income, mean_spending):
    """Return suggestion string based on income and spending profile."""
    income = profile['Annual Income (k$)']
    spending = profile['Spending Score (1-100)']

    if income > mean_income and spending > mean_spending:
        return "High Income, High Spending"
    elif income > mean_income and spending <= mean_spending:
        return "High Income, Low Spending"
    elif income <= mean_income and spending > mean_spending:
        return "Low Income, High Spending"
    else:
        return "Low Income, Low Spending"

def show_detailed_overview(df):
    """Display detailed visualizations of the dataset."""
    st.write("## Detailed Dataset Overview")
    
    # 1. Gender Distribution
    if 'Gender' in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            gender_counts = df['Gender'].value_counts()
            fig_gender_bar = px.bar(
                x=gender_counts.index, 
                y=gender_counts.values,
                title='Gender Distribution',
                labels={'x': 'Gender', 'y': 'Count'}
            )
            st.plotly_chart(fig_gender_bar, use_container_width=True)
        
        with col2:
            fig_gender_pie = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title='Gender Distribution Ratio'
            )
            st.plotly_chart(fig_gender_pie, use_container_width=True)

    # 2. Age Distribution
    st.write("### Age Analysis")
    fig_age = px.histogram(
        df, x='Age',
        title='Age Distribution',
        nbins=20
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # 3. Annual Income Distribution
    st.write("### Income Analysis")
    fig_income = px.histogram(
        df, x='Annual Income (k$)',
        title='Annual Income Distribution',
        nbins=20
    )
    st.plotly_chart(fig_income, use_container_width=True)

    # 4. Spending Score Distribution
    st.write("### Spending Score Analysis")
    fig_spending = px.histogram(
        df, x='Spending Score (1-100)',
        title='Spending Score Distribution',
        nbins=20
    )
    st.plotly_chart(fig_spending, use_container_width=True)

    # 5. Age vs Annual Income
    st.write("### Relationship Analysis")
    if 'Gender' in df.columns:
        fig_age_income = px.scatter(
            df, 
            x='Age', 
            y='Annual Income (k$)',
            color='Gender',
            title='Age vs Annual Income'
        )
    else:
        fig_age_income = px.scatter(
            df, 
            x='Age', 
            y='Annual Income (k$)',
            title='Age vs Annual Income'
        )
    st.plotly_chart(fig_age_income, use_container_width=True)

    # 6. Elbow Method
    st.write("### Clustering Analysis")
    X = df[REQUIRED_COLUMNS].values
    inertias = []
    K = range(1, 11)
    
    with st.spinner('Calculating Elbow Curve...'):
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
    
    fig_elbow = px.line(
        x=list(K), y=inertias,
        title='Elbow Method for Optimal K',
        labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'}
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

def main():
    st.title("Mall Customers Clustering with 3D Visualization")

    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload your customer CSV file", type=['csv'])

    df = None # Original DataFrame to store data for display and plotting
    X_scaled_for_clustering = None # Scaled NumPy array specifically for KMeans input

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if not validate_columns(df):
            st.error(f"Uploaded file must contain the following columns: {REQUIRED_COLUMNS}")
            st.stop() # Stop execution if columns are missing
        # If a file is uploaded, scale its features for clustering
        X_scaled_for_clustering = scale_features(df, REQUIRED_COLUMNS)
    else:
        st.warning("No file uploaded. You can choose to use the default dataset.")
        fallback_choice = st.radio(
            "Do you want to use the default Mall Customers dataset?",
            ("No", "Yes"),
            index=0
        )
        if fallback_choice == "Yes":
            try:
                # Call the actual load_and_scale_data function from notebooks.data_utils
                # This function returns the original DataFrame and the scaled NumPy array
                df, X_scaled_for_clustering = load_and_scale_data()
            except FileNotFoundError:
                st.error("Default dataset 'Mall_Customers.csv' not found. "
                         "Please ensure it's located at `your_project_root/data/Mall_Customers.csv` "
                         "and your `app.py` is at `your_project_root/app.py` and `data_utils.py` at `your_project_root/notebooks/data_utils.py`.")
                st.stop()
            except Exception as e:
                st.error(f"An unexpected error occurred while loading the default data: {e}")
                st.stop()
        else:
            st.info("Please upload a CSV file or choose to use the default dataset to proceed.")
            st.stop() # Stop execution if neither option is taken

    # Proceed with processing only if data has been successfully loaded/uploaded
    if df is not None and X_scaled_for_clustering is not None:
        k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=5)
        cluster_labels = perform_clustering(X_scaled_for_clustering, k)
        # Add the cluster labels to the original DataFrame for display and plotting
        df[CLUSTER_COL] = cluster_labels.astype(str)

        fig = plot_3d_clusters(df, REQUIRED_COLUMNS, CLUSTER_COL)
        st.plotly_chart(fig, use_container_width=True)

        cluster_profiles = get_cluster_profiles(df, CLUSTER_COL, REQUIRED_COLUMNS)
        st.write("### Cluster Profiles (Mean Values):")
        st.dataframe(cluster_profiles)

        st.write("### Customer Type Suggestions:")
        mean_income = df['Annual Income (k$)'].mean()
        mean_spending = df['Spending Score (1-100)'].mean()

        for cluster in cluster_profiles.index:
            profile = cluster_profiles.loc[cluster]
            suggestion = suggest_customer_type(profile, mean_income, mean_spending)
            st.write(f"**Cluster {cluster}:** {suggestion}")

        csv = df.to_csv(index=False).encode()
        st.download_button(
            label="Download clustered data CSV",
            data=csv,
            file_name="clustered_customers.csv",
            mime="text/csv"
        )

    if df is not None and X_scaled_for_clustering is not None:
        # Add this section before the clustering controls
        if st.button("Show Detailed Overview"):
            show_detailed_overview(df)
        
        st.write("---")  # Add a separator
        
if __name__ == "__main__":
    main()
