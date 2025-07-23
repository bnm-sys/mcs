import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from notebooks.data_utils import load_and_scale_data


REQUIRED_COLUMNS = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
CLUSTER_COL = 'Cluster'


def validate_columns(df):
    """Ensure required columns are present in the uploaded CSV."""
    return all(col in df.columns for col in REQUIRED_COLUMNS)


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

    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload your customer CSV file", type=['csv'])

    use_default = False
    df = None

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if not validate_columns(df):
            st.error(f"Uploaded file must contain the following columns: {REQUIRED_COLUMNS}")
            return
    else:
        st.warning("No file uploaded.")
        fallback_choice = st.radio(
            "Do you want to use the default Mall Customers dataset?",
            ("No", "Yes"),
            index=0
        )
        if fallback_choice == "Yes":
            df, _ = load_and_scale_data()
            use_default = True
        else:
            st.stop()

    # Proceed with processing
    X_scaled = scale_features(df, REQUIRED_COLUMNS)

    k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=5)
    cluster_labels = perform_clustering(X_scaled, k)
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
