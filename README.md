# MCS : Mall Customer Segmentation 

## Overview
MCS is an interactive web application built with Streamlit that performs customer segmentation analysis using K-means clustering. The application provides detailed 3D visualization of customer clusters and comprehensive data analysis tools.

## Web Address
We have an streamlit app up and running at most time, check it out at [Our MCS App](http://mcs-bdp.streamlit.app/)
## Features
- **Interactive Data Upload**: Upload custom customer data or use the default Mall Customers dataset
- **3D Cluster Visualization**: Dynamic 3D scatter plots showing customer segments
- **Detailed Data Analysis**:
  - Gender Distribution Analysis (bar plots and pie charts)
  - Age Distribution Analysis
  - Annual Income Distribution
  - Spending Score Analysis
  - Age vs. Income Relationship Analysis
  - Elbow Method for Optimal Cluster Selection
- **Customer Segmentation**: Automated K-means clustering with adjustable cluster numbers
- **Cluster Profiling**: Detailed analysis of each cluster's characteristics
- **Data Export**: Download capabilities for clustered data

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI_VISEM.git
cd AI_VISEM
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```
AI_VISEM/
├── app.py              # Main Streamlit application
├── notebooks/
│   └── data_utils.py   # Data loading and preprocessing utilities
├── data/
│   └── Mall_Customers.csv  # Default dataset
└── requirements.txt    # Project dependencies
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the application in your web browser (typically at `http://localhost:8501`)

3. Either:
   - Upload your own CSV file containing customer data
   - Use the default Mall Customers dataset

4. Required CSV columns:
   - Age
   - Annual Income (k$)
   - Spending Score (1-100)
   - Gender (optional)

## Data Analysis Features

### Cluster Analysis
- Adjustable number of clusters (K) using a slider
- 3D visualization of clusters
- Detailed cluster profiles with mean values
- Customer type suggestions based on income and spending patterns

### Detailed Overview
- Gender distribution visualization (if gender data is available)
- Age distribution analysis
- Income distribution analysis
- Spending score patterns
- Relationship analysis between variables
- Elbow method visualization for optimal K selection

## Export Options
- Download clustered data as CSV
- Interactive plots with zoom, pan, and save capabilities

## Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- plotly

## License
Liscensed under CC by Binam Poudel.

## Contributors
- Binam Poudel
- Dikshya Rai
- Pranisha Poudel

## Acknowledgments
- Mall Customer Dataset (Kaggle)
- Streamlit Community
- Python Data Science Community