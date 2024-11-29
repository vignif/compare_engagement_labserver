import pandas as pd
import numpy as np
import optuna
import argparse
from pathlib import Path

class FindParams:
    def __init__(self, file_path, n_trials=1000):
        self.file_path = file_path
        self.dataframe = None
        self.get_dataframe()
        self.n_trials=n_trials
        print(f"Finding {self.file_path} with {self.n_trials} max trials.")

    def get_dataframe(self):
        """Load and return the DataFrame."""
        try:
            self.dataframe = pd.read_csv(self.file_path, sep=';')
        except Exception as e:
            raise ValueError(f"Error loading DataFrame: {e}")

    def error_function(self, trial):
        """Calculate the mean squared error over the entire dataset using Optuna."""
        prox_epsilon = trial.suggest_uniform('prox_epsilon', self.dataframe['prox_epsilon'].min(), self.dataframe['prox_epsilon'].max())
        prox_weight = trial.suggest_uniform('prox_weight', 0, 1)
        gaze_weight = trial.suggest_uniform('gaze_weight', 1 - prox_weight, 1 - prox_weight)
        
        
        # Calculate distances for all rows
        distances = np.abs(self.dataframe['prox_epsilon'] - prox_epsilon) + \
                    np.abs(self.dataframe['prox_weight'] - prox_weight) + \
                    np.abs(self.dataframe['gaze_weight'] - gaze_weight)
        
        # Assign weights to each row based on the distance
        weights = 1 / (distances + 1e-6)
        
        # Calculate the weighted mean squared error
        weighted_mse = np.sum(weights * ((self.dataframe['eng_1'] - self.dataframe['grace_eng']) ** 2)) / np.sum(weights)
        
        return weighted_mse

    def optimize_parameters(self):
        """Optimize prox_epsilon, prox_weight, and gaze_weight to minimize the error using Optuna."""
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self.error_function, n_trials=self.n_trials)
        
        best_params = study.best_params
        best_mse = study.best_value
        
        print("Optimal parameters found:")
        print(f"prox_epsilon: {best_params['prox_epsilon']}")
        print(f"prox_weight: {best_params['prox_weight']}")
        print(f"gaze_weight: {best_params['gaze_weight']}")
        print(f"Overall weighted MSE at optimal parameters: {best_mse}")
        
        return best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find params')
    parser.add_argument('--file_path', type=str, help='Path to the CSV file', default="csv/dataset.csv")
    parser.add_argument('--n_trials', type=int, default=1000)
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    n_trials = args.n_trials
    
    fp = FindParams(file_path, n_trials)
    try:
        optimal_params = fp.optimize_parameters()
    except ValueError as e:
        print(e)
