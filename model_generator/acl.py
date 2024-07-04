import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances, accuracy_score
from itertools import combinations, product
from sklearn.utils import shuffle

# Step-1: Isolating Important Patterns
def isolate_important_patterns(model, data, predictions, target_class):
    important_patterns = data[predictions == target_class]
    return important_patterns

# Step-2: Generate Additional Test Instances
def generate_t_way_tests(ipm, t=2):
    all_combinations = []
    for combination in combinations(ipm.values(), t):
        all_combinations.extend(product(*combination))
    return pd.DataFrame(all_combinations, columns=ipm.keys())

def refine_test_instances(ipm, important_patterns):
    additional_tests = generate_t_way_tests(ipm)
    refined_tests = additional_tests.loc[~additional_tests.isin(important_patterns).all(axis=1)]
    return refined_tests

# Step-3a: Uncertainty-Based Strategy
def calculate_entropy(probabilities):
    return -np.sum(probabilities * np.log(probabilities), axis=1)

# Step-3b: Diversity-Based Strategy
def calculate_diversity(data):
    distances = pairwise_distances(data)
    diversity_scores = np.min(distances + np.eye(len(distances)) * distances.max(), axis=1)
    return diversity_scores

# Step-3c: Combined Strategy
def rank_instances(model, data, alpha=0.5):
    probabilities = model.predict_proba(data)
    entropies = calculate_entropy(probabilities)
    diversities = calculate_diversity(data)
    combined_scores = alpha * entropies + (1 - alpha) * diversities
    ranked_indices = np.argsort(combined_scores)[::-1]  # Highest scores first
    return data.iloc[ranked_indices]

# Function to run Phase-2
def run_phase_2(original_model, data, target, ipm, iterations=5, alpha=0.5, top_k=10):
    surrogate_data = pd.DataFrame()
    surrogate_labels = []

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")
        
        # Predict class labels using the original model
        predictions = original_model.predict(data)

        for class_label in [0, 1]:
            print(f"Processing class label: {class_label}")
            
            # Step-1: Isolate Important Patterns
            important_patterns = isolate_important_patterns(original_model, data, predictions, class_label)
            
            # Step-2: Generate Additional Test Instances
            additional_tests = refine_test_instances(ipm, important_patterns)
            
            # Step-3: Active Learning for Additional Data Points
            # Combine uncertainty and diversity strategies
            informative_tests = rank_instances(original_model, additional_tests, alpha)[:top_k]

            # Collect data for the surrogate model
            surrogate_data = pd.concat([surrogate_data, informative_tests])
            surrogate_labels.extend([class_label] * len(informative_tests))
    
    # Train surrogate model
    surrogate_model = RandomForestClassifier()
    surrogate_model.fit(surrogate_data, surrogate_labels)
    
    accuracy = accuracy_score(surrogate_labels, surrogate_model.predict(surrogate_data))
    print(f'Surrogate model accuracy: {accuracy}')
    
    return surrogate_model, accuracy
