import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from data_prep import load_and_split_data
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, validation_df, optimal_params):
        self.validation_df = validation_df
        self.optimal_params = optimal_params
        self.filtered_df = None

    def evaluate(self):
        prox_weight = self.optimal_params['prox_weight']
        gaze_weight = self.optimal_params['gaze_weight']
        prox_epsilon = self.optimal_params['prox_epsilon']
        threshold = 0.05  # Adjust this value based on your tolerance for "closeness"

        filtered_df = self.validation_df[
            (np.abs(self.validation_df['prox_epsilon'] - prox_epsilon) < threshold) &
             (np.abs(self.validation_df['prox_weight'] - prox_weight) < threshold) &
            (np.abs(self.validation_df['gaze_weight'] - gaze_weight) < threshold)
        ]
        # filtered_df = self.validation_df
        # Check if we have any rows left after filtering
        if filtered_df.empty:
            print("No rows found that match the criteria.")
            return
        self.filtered_df = filtered_df
        # Compare the engagement values
        print(f"Number of filtered rows: {len(filtered_df)}")
        return self.compare_engagement(filtered_df)
    
    
    def search_prox(self):
        # Get optimal parameters
        prox_epsilon = self.optimal_params['prox_epsilon']

        # Define a threshold for how close parameters should be considered "like" the optimal ones
        threshold = 0.001  # Adjust this value based on your tolerance for "closeness"

        filtered_df = self.validation_df[
            (np.abs(self.validation_df['prox_epsilon'] - prox_epsilon) < threshold)
        ]
        # filtered_df = self.validation_df
        # Check if we have any rows left after filtering
        if filtered_df.empty:
            print("No rows found that match the criteria.")
            return
        self.filtered_df = filtered_df
        # Compare the engagement values
        print(f"Number of filtered rows: {len(filtered_df)}")
        return self.compare_engagement(filtered_df)

    def search_weights(self):
                # Get optimal parameters
        prox_weight = self.optimal_params['prox_weight']
        gaze_weight = self.optimal_params['gaze_weight']

        # Define a threshold for how close parameters should be considered "like" the optimal ones
        threshold = 0.001  # Adjust this value based on your tolerance for "closeness"

        # Filter the validation set based on proximity to the optimal parameters
        filtered_df = self.validation_df[
            (np.abs(self.validation_df['prox_weight'] - prox_weight) < threshold) &
            (np.abs(self.validation_df['gaze_weight'] - gaze_weight) < threshold)
        ]

        # filtered_df = self.validation_df
        # Check if we have any rows left after filtering
        if filtered_df.empty:
            print("No rows found that match the criteria.")
            return
        self.filtered_df = filtered_df
        # Compare the engagement values
        print(f"Number of filtered rows: {len(filtered_df)}")
        return self.compare_engagement(filtered_df)

    def compare_engagement(self, filtered_df):
        # Calculate mean engagement values for comparison
        mean_eng_1 = filtered_df['eng_1'].mean()
        mean_grace_eng = filtered_df['grace_eng'].mean()
        mse = ((filtered_df['eng_1'] - filtered_df['grace_eng']) ** 2).mean()

        # print(f"Mean engagement (eng_1): {mean_eng_1}")
        # print(f"Mean engagement (grace_eng): {mean_grace_eng}")

        # val = mse
        # print(f"Eng1 / grace: {val}")
        # Optionally plot the results
        # self.plot_results(filtered_df)
        # return val

        print(f"mse: {mse}")
        return mse

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.filtered_df['eng_1'], self.filtered_df['grace_eng'], alpha=0.5)
        plt.title('Filtered Engagement Comparison')
        plt.xlabel('eng_1 (Actual)')
        plt.ylabel('grace_eng (Predicted)')
        plt.plot([0, 1], [0, 1], 'r--')  # Line for perfect predictions
        plt.grid()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

def make_plot():
    # Load and split the dataset
    val_df, _ = load_and_split_data('dataset.csv')  # Load only the validation part

    # Load optimal parameters from results.json
    with open('results.json') as f:
        results = json.load(f)
        optimal_params = results['best_params']
    optimal_params['prox_epsilon'] = 1.5139999
    optimal_params['prox_weight'] = 0.03
    optimal_params['gaze_weight'] = 0.97
        
    evaluator = ModelEvaluator(val_df, optimal_params)
    evaluation_result = evaluator.evaluate()  # Assume evaluate() returns some results
    evaluator.plot_results()

def search_weights():
     # Load and split the dataset
    val_df, _ = load_and_split_data('dataset.csv')  # Load only the validation part

    optimal_params = {}
    optimal_params['prox_epsilon'] = 1.5139999
    optimal_params['prox_weight'] = 0.4
    optimal_params['gaze_weight'] = 0.6

    # Assuming optimal_params and val_df are already defined
    closest_value = None
    closest_distance = float('inf')  # Start with a very large distance
    session_data = {}  # Dictionary to store the session information

    for a in tqdm(np.arange(0.01, 1, 0.001), desc="Optimizing Prox Epsilon"):
        optimal_params['prox_weight'] = a
        optimal_params['gaze_weight'] = 1 - optimal_params['prox_weight']
        print(f"Epsilon {a}")
        
        evaluator = ModelEvaluator(val_df, optimal_params)
        evaluation_result = evaluator.search_weights()  # Assume evaluate() returns some results
        if not isinstance(evaluation_result, float):
            continue
        # Check how close `a` is to 1.00
        distance = abs(evaluation_result - 1.00)
        if distance < closest_distance:
            closest_distance = distance
            closest_value = a
            session_data = {
                'prox_weight': a,
                'evaluation_result': evaluation_result,  # Store the evaluation results
                'distance_to_one': closest_distance
            }

    # After the loop, you can print or save the session data
    print(f"Closest prox_weight to 1.00: {closest_value}")
    print(f"Session Data: {session_data}")

def search_prox():
    # Load and split the dataset
    _, val_df = load_and_split_data('dataset.csv')  # Load only the validation part

    # Load optimal parameters from results.json
    with open('results.json') as f:
        results = json.load(f)
        optimal_params = results['best_params']
    optimal_params['prox_epsilon'] = 1.5
    optimal_params['prox_weight'] = 0.4
    optimal_params['gaze_weight'] = 0.6

    # Assuming optimal_params and val_df are already defined
    closest_value = None
    closest_distance = float('inf')  # Start with a very large distance
    session_data = {}  # Dictionary to store the session information

    for a in tqdm(np.arange(0.01, 1, 0.001), desc="Optimizing Prox Epsilon"):
        optimal_params['prox_epsilon'] = a
        print(f"Epsilon {a}")
        
        evaluator = ModelEvaluator(val_df, optimal_params)
        evaluation_result = evaluator.evaluate()  # Assume evaluate() returns some results
        
        # Check how close `a` is to 1.00
        distance = abs(evaluation_result - 1.00)
        if distance < closest_distance:
            closest_distance = distance
            closest_value = a
            session_data = {
                'prox_epsilon': a,
                'evaluation_result': evaluation_result,  # Store the evaluation results
                'distance_to_one': closest_distance
            }

    # After the loop, you can print or save the session data
    print(f"Closest prox_epsilon to 1.00: {closest_value}")
    print(f"Session Data: {session_data}")


def main():
     # Load and split the dataset
    _, val_df = load_and_split_data('dataset.csv')  # Load only the validation part

    # Load optimal parameters from results.json
    with open('results4.json') as f:
        results = json.load(f)
        optimal_params = results['best_params']
    print("Input params")
    print(optimal_params)
    evaluator = ModelEvaluator(val_df, optimal_params)
    evaluation_result = evaluator.evaluate()  # Assume evaluate() returns some results
    print()

if __name__ == "__main__":
    # search_prox()
    # search_weights()
    # make_plot()
    main()