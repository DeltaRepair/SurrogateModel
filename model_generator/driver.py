# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd  
import numpy as np
# Tensorflow import
import tensorflow as tf

#local Iports
from utils import helpers
from data import Data
from model import Model
from dice import Dice

from cacheConfig import OutcomeCache
from cacheConfigTest import oc_test
from dd import DD

from dd import DD
import pandas as pd  
from covering_array import*
from funcs import*
from data import*
from model import*
from dice import*
from descritization import* 

# Generate t-way test cases
generate_t_way_tests(features, t=2):
    all_combinations = []
    for combination in itertools.combinations(features, t):
        all_combinations.extend(itertools.product(*combination))
    return all_combinations

# Create Input Parameter Model (IPM)
create_ipm(data):
    ipm = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            ipm[column] = data[column].unique()
        else:
            ipm[column] = np.linspace(data[column].min(), data[column].max(), num=10)
    return ipm

generate_counterfactuals(numberOfCFs)
    d = dice_ml.Data(dataframe=train_dataset,
                     continuous_features=['age', 'hours_per_week'],
                     outcome_name='income')
    
    
    m = dice_ml.Model(model_path=dice_ml.utils.helpers.get_adult_income_modelpath(),
                      backend='TF2', func="ohe-min-max")
    # DiCE explanation instance
    return dice_ml.Dice(d,m)

def construct_surrogate_model(original_model, data, target):
    # Phase 1: Generating t-way Tests
    ipm = create_ipm(data)
    initial_tests = generate_additional_tests(ipm, data)
    initial_tests = shuffle(initial_tests).reset_index(drop=True)
    
    # Phase 2: Refining and Generating Additional Test Cases
    surrogate_data = pd.DataFrame()
    surrogate_labels = []
    for i in range(2):
        initial_predictions = original_model.predict(initial_tests)
        for class_label in [0, 1]:
            class_data = initial_tests[initial_predictions == class_label]
            additional_tests = generate_additional_tests(ipm, class_data)
            informative_tests = select_informative_instances(original_model, additional_tests)
            surrogate_data = pd.concat([surrogate_data, informative_tests])
            surrogate_labels.extend([class_label] * len(informative_tests))
    
    # Handling Insufficient Opposite Class Prediction
    for i in range(2):
        counterfactuals = []
        for instance in surrogate_data[surrogate_labels == i].values:
            counterfactual = generate_counterfactual(instance, original_model, ipm)
            counterfactuals.append(counterfactual)
        counterfactuals_df = pd.DataFrame(counterfactuals, columns=surrogate_data.columns)
        surrogate_data = pd.concat([surrogate_data, counterfactuals_df])
        surrogate_labels.extend([1 - i] * len(counterfactuals))
    
    # Training surrogate model
    surrogate_model = RandomForestClassifier()
    surrogate_model.fit(surrogate_data, surrogate_labels)
    
    return surrogate_model, accuracy_score(surrogate_labels, surrogate_model.predict(surrogate_data))

