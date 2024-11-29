import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm  # Progress bar
import json


class FindParams:
    def __init__(self, file_path, threshold):
        """
        Initialize the class with file path and threshold.

        Parameters:
        - file_path (str): Path to the CSV file containing the dataset.
        - threshold (float): Value for filtering based on a specific column difference.
        """
        self.file_path = file_path
        self.df = None
        self.threshold = threshold
        self.get_dataframe()
        self.len = self.df.shape[0]
        self.params = {"prox_eps": [], "prox_weight": [], "gaze_weight": []}
        print(f"Running with threshold: {self.threshold}")
        

    def get_dataframe(self):
        """Load and return the DataFrame."""
        try:
            self.df = pd.read_csv(self.file_path, sep=';')
        except Exception as e:
            raise ValueError(f"Error loading DataFrame: {e}")
    
    def execute(self):
        """
        Process the DataFrame and find optimal parameters based on the threshold.
        """
        prox_eps = []
        gaze_weights = []
        prox_weights = []
        for index, row in tqdm(self.df.iterrows(), total=self.len, desc="Processing rows"):

            # val = np.abs(row["grace_eng"] - row["eng_1"])
            val = np.sqrt((row["grace_eng"] - row["eng_1"]) ** 2)
            if val < self.threshold:
                prox_eps.append(row["prox_epsilon"])
                gaze_weights.append(row["gaze_weight"])
                prox_weights.append(row["prox_weight"])

        # Store parameters
        self.params["prox_eps"] = prox_eps
        self.params["prox_weight"] = prox_weights
        self.params["gaze_weight"] = gaze_weights
        
        # Print optimal parameters
        print(f"Optimal parameters found for threshold {self.threshold}:")
        print(f"prox_epsilon: {np.mean(self.params['prox_eps']):.3f} (STD: {np.std(self.params['prox_eps']):.3f})")
        print(f"prox_weight: {np.mean(self.params['prox_weight']):.3f} (STD: {np.std(self.params['prox_weight']):.3f})")
        print(f"gaze_weight: {np.mean(self.params['gaze_weight']):.3f} (STD: {np.std(self.params['gaze_weight']):.3f})")
        print(f"Used: {len(self.params['prox_eps']) / self.len:.2%} of the dataset\n")

        # Return the results to be accumulated
        return {
            "threshold": self.threshold,
            "used_data": len(self.params['prox_eps']) / self.len,
            "prox_epsilon": {
                "mean": round(np.mean(self.params['prox_eps']), 3),
                "std": round(np.std(self.params['prox_eps']), 3)
            },
            "prox_weight": {
                "mean": round(np.mean(self.params['prox_weight']), 3),
                "std": round(np.std(self.params['prox_weight']), 3)
            },
            "gaze_weight": {
                "mean": round(np.mean(self.params['gaze_weight']), 3),
                "std": round(np.std(self.params['gaze_weight']), 3)
            }
        }

    
def save_all_results(results, output_file):
    """
    Save all results from multiple executions into a single JSON file.

    Parameters:
    - results (list): List of dictionaries containing the results for each threshold.
    - output_file (str): The name of the output file.
    """
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"All results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find optimal parameters')
    parser.add_argument('--file_path', type=str, help='Path to the CSV file', default="dataset.csv")
    parser.add_argument('--output_file', type=str, help='Path to save the aggregated results', default="aggregated_results.json")
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    output_file = args.output_file

    all_results = []  # To accumulate results from all runs
    # Parameters for the beta distribution
    # a = 0.1  # First shape parameter
    # b = 2.0  # Second shape parameter
    # size = 10000  # Total number of values to generate

    # # Generate beta distributed values
    # beta_values = np.random.beta(a, b, size)

    # # Scale to the desired range (0.01 to 1.0)
    # scaled_beta_values = beta_values * (1.0 - 0.01) + 0.01

    scaled_beta_values = np.arange(0.1,1,0.1)    
    try:
        # Loop through different threshold values
        for th in scaled_beta_values:
            fp = FindParams(file_path, th)
            result = fp.execute()  # Execute and collect the result for each threshold
            all_results.append(result)  # Accumulate results
            
        # Save all the accumulated results into one file
        save_all_results(all_results, output_file)

    except ValueError as e:
        print(e)
