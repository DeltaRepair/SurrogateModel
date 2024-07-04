import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_ipm(data, num_bins=10):
    """
    Create an Input Parameter Model (IPM) using uniform sampling for numerical features 
    and unique values for categorical features.
    
    Parameters:
        data (pd.DataFrame): The input tabular data.
        num_bins (int): The number of bins to use for numerical features.
        
    Returns:
        ipm (dict): A dictionary where keys are feature names and values are lists of representative values for each feature.
    """
    ipm = {}
    
    for column in data.columns:
        if data[column].dtype == 'object' or data[column].dtype.name == 'category':
            # For categorical features, use unique values
            ipm[column] = data[column].unique().tolist()
        else:
            # For numerical features, use uniform sampling to create bins
            min_val = data[column].min()
            max_val = data[column].max()
            bins = np.linspace(min_val, max_val, num_bins)
            ipm[column] = bins.tolist()
    
    return ipm

# Example usage
data = pd.DataFrame({
    'feature1': ['a', 'b', 'a', 'c'],
    'feature2': [1.2, 3.4, 2.2, 4.5],
    'feature3': [10, 20, 15, 25]
})

ipm = create_ipm(data)
print(ipm)
