# Mall Customers Clustering Project

## Overview

This project performs customer segmentation on the Mall Customers dataset using K-Means clustering. The goal is to identify distinct groups of customers based on Age, Annual Income, and Spending Score to help businesses tailor marketing strategies effectively.

## Project Structure

- `data/`  
  Contains the dataset file: `Mall_Customers.csv`

- `notebooks/`  
  Jupyter notebooks for each step:  
  - `0_data_exploration.ipynb` — Initial data exploration and statistics  
  - `1_preprocessing.ipynb` — Data preprocessing and feature scaling  
  - `2_elbow_method.ipynb` — Finding optimal cluster count with the Elbow Method  
  - `3_clustering.ipynb` — Applying K-Means clustering  
  - `4_visualization.ipynb` — Visualizing clusters in 2D and 3D  
  - `5_cluster_analysis.ipynb` — Analyzing clusters and identifying customer types  
  - `data_utils.py` — Utility functions for loading and scaling data

- `outputs/`  
  Folder for saving results such as `clustered_customers.csv` with cluster labels

## Key Libraries Used

- `pandas` — Data manipulation  
- `scikit-learn` — Data preprocessing and clustering  
- `matplotlib` & `seaborn` — Data visualization

## How to Run

1. Ensure you have Python 3.x installed along with the required packages (see `requirements.txt`).  
2. Open and run the notebooks sequentially in Jupyter Notebook or JupyterLab.  
3. The final output file `clustered_customers.csv` will be saved inside the `outputs/` folder.

## Summary of Findings

- Five distinct customer clusters were identified based on income and spending behavior.  
- Customer types include High Income - High Spending, High Income - Low Spending, Low Income - High Spending, and Low Income - Low Spending.  
- These insights can help tailor marketing campaigns and improve customer engagement.
