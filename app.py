import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from notebooks.data_utils import load_and_scale_data


def load_data():
    """Load raw data and return original DataFrame."""
    df, _ = load_and_scale_data()
    return df


def scale_features(df, features):
    """Scale features using StandardScaler and return scaled numpy array."""
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


def main():
    st.title("Mall Customers Clustering with 3D Visualization")

    # Constants
    FEATURES = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    CLUSTER_COL = 'Cluster'

    # Load and prepare data
    df = load_data()
    X_scaled = scale_features(df, FEATURES)

    # Select number of clusters with slider
    k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=5)

    # Perform clustering
    cluster_labels = perform_clustering(X_scaled, k)
    df[CLUSTER_COL] = cluster_labels.astype(str)  # For categorical coloring

    # Plot interactive 3D clusters
    fig = plot_3d_clusters(df, FEATURES, CLUSTER_COL)
    st.plotly_chart(fig, use_container_width=True)

    # Show cluster profiles
    cluster_profiles = get_cluster_profiles(df, CLUSTER_COL, FEATURES)
    st.write("### Cluster Profiles (Mean Values):")
    st.dataframe(cluster_profiles)

    # Show customer type suggestions
    st.write("### Customer Type Suggestions:")
    mean_income = df['Annual Income (k$)'].mean()
    mean_spending = df['Spending Score (1-100)'].mean()

    for cluster in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster]
        suggestion = suggest_customer_type(profile, mean_income, mean_spending)
        st.write(f"**Cluster {cluster}:** {suggestion}")

    # CSV download button
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="Download clustered data CSV",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
