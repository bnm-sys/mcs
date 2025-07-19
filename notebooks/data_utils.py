# notebooks/data_utils.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_scale_data():
    """
    Loads the Mall Customers dataset and scales selected features.

    Returns:
    - df (pd.DataFrame): Original dataset with all columns.
    - X_scaled_df (pd.DataFrame): Scaled features as a DataFrame with columns ['Age', 'Annual Income', 'Spending Score'].
    """

    # Dynamically find the root of the project
    notebook_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
    csv_path = os.path.join(project_root, 'data', 'Mall_Customers.csv')

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Select and scale features
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=['Age', 'Annual Income', 'Spending Score'])

    return df, X_scaled_df
