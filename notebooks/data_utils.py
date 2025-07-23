# notebooks/data_utils.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_scale_data():
    """
    Loads the Mall Customers dataset and scales selected features.

    Returns:
    - df (pd.DataFrame): Original dataset with all columns.
    - X_scaled_array (np.ndarray): Scaled features as a NumPy array.
    """
    # Define REQUIRED_COLUMNS here to ensure consistency with app.py
    REQUIRED_COLUMNS = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    # Dynamically find the root of the project
    # Assuming app.py is at project_root/app.py
    # and data_utils.py is at project_root/notebooks/data_utils.py
    # and Mall_Customers.csv is at project_root/data/Mall_Customers.csv
    notebook_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(notebook_dir, '..')) # Go up from 'notebooks'
    csv_path = os.path.join(project_root, 'data', 'Mall_Customers.csv')

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Select and scale features
    features_to_scale = REQUIRED_COLUMNS
    X = df[features_to_scale]
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X) # Return as numpy array directly for clustering

    return df, X_scaled_array