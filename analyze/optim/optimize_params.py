import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import json
from data_prep import load_and_split_data  # Custom function to split your dataset

class FindParams:
    """
    A class to optimize the parameters `prox_epsilon`, `prox_weight`, and `gaze_weight`
    using Optuna to minimize the weighted mean squared error (MSE).
    """

    def __init__(self, df):
        """
        Initializes the FindParams class with the dataset.

        Args:
            df (pd.DataFrame): The dataset for optimization.
        """
        self.dataframe = df
        self.feature_min_max = self._precompute_feature_ranges()

    def _precompute_feature_ranges(self):
        """
        Precomputes the min and max for features `prox_epsilon`, `prox_weight`, and `gaze_weight`.

        Returns:
            dict: A dictionary with min-max ranges for each feature.
        """
        return {
            'prox_epsilon': (self.dataframe['prox_epsilon'].min(), self.dataframe['prox_epsilon'].max()),
            'prox_weight': (0, 1),
            'gaze_weight': (0, 1)
        }

    def _error_function(self, trial):
        """
        Objective function for Optuna to minimize the weighted MSE.

        Args:
            trial (optuna.trial.Trial): A trial object suggesting parameter values.

        Returns:
            float: The weighted mean squared error.
        """
        # Suggest parameter values based on precomputed ranges
        prox_epsilon = trial.suggest_uniform('prox_epsilon', *self.feature_min_max['prox_epsilon'])
        prox_weight = trial.suggest_uniform('prox_weight', *self.feature_min_max['prox_weight'])
        gaze_weight = trial.suggest_uniform('gaze_weight', *self.feature_min_max['gaze_weight'])

        # Normalize weights to ensure their sum does not exceed 1 (optional constraint)
        weight_sum = prox_weight + gaze_weight
        if weight_sum > 1:
            prox_weight /= weight_sum
            gaze_weight /= weight_sum

        # Calculate distances and weights
        distances = np.abs(self.dataframe['prox_epsilon'] - prox_epsilon) + \
                    np.abs(self.dataframe['prox_weight'] - prox_weight) + \
                    np.abs(self.dataframe['gaze_weight'] - gaze_weight)
        weights = 1 / (distances + 1e-6)  # Add small epsilon to avoid division by zero

        # Compute weighted mean squared error
        weighted_mse = np.sum(weights * ((self.dataframe['eng_1'] - self.dataframe['grace_eng']) ** 2)) / np.sum(weights)
        return weighted_mse

    def optimize_parameters(self, n_trials=3000, patience=1000, min_delta=1e-5, seed=42, results_file='results.json'):
        """
        Optimize `prox_epsilon`, `prox_weight`, and `gaze_weight` to minimize weighted MSE.

        Args:
            n_trials (int): Maximum number of trials.
            patience (int): Number of trials to wait for improvement before stopping.
            min_delta (float): Minimum change in MSE to count as improvement.
            seed (int): Random seed for reproducibility.
            results_file (str): File path to save the results.

        Returns:
            dict: The best parameter values found.
        """
        def early_stopping_callback(study, trial):
            if len(study.trials) > patience:
                last_trials = study.trials[-patience:]
                if all(np.abs(trial.value - last_trials[0].value) < min_delta for trial in last_trials):
                    study.stop()

        # Create an Optuna study with SQLite storage for persistence
        sampler = TPESampler(seed=seed)
        storage = "sqlite:///optuna_study.db"
        study = optuna.create_study(direction='minimize', storage=storage, load_if_exists=True, sampler=sampler)

        # Optimize with early stopping
        study.optimize(self._error_function, n_trials=n_trials, callbacks=[early_stopping_callback])

        # Extract and save the results
        best_params = study.best_params
        best_mse = study.best_value
        self._save_results(best_params, best_mse, study.trials, results_file)

        # Print the best parameters and their corresponding MSE
        print(f"Optimal parameters:\nprox_epsilon: {best_params['prox_epsilon']}\n"
              f"prox_weight: {best_params['prox_weight']}\n"
              f"gaze_weight: {best_params['gaze_weight']}\nWeighted MSE: {best_mse}")
        return best_params

    def _save_results(self, best_params, best_mse, trials, results_file):
        """
        Saves the optimization results and all trial data to a file.

        Args:
            best_params (dict): The best parameters found.
            best_mse (float): The corresponding MSE.
            trials (list of optuna.trial.FrozenTrial): The list of all trials.
            results_file (str): Path to save the results.
        """
        results_data = {
            'best_params': best_params,
            'best_mse': best_mse,
            'trials': [
                {
                    'trial_number': trial.number,
                    'params': trial.params,
                    'value': trial.value
                }
                for trial in trials
            ]
        }
        # Save results as JSON
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {results_file}")

def main():
    """
    Main function to run the parameter optimization.
    """
    # Load and split the dataset
    train_df, _ = load_and_split_data('dataset.csv')  # Use only training data

    # Initialize optimizer and run
    fp = FindParams(train_df)
    fp.optimize_parameters()

if __name__ == "__main__":
    main()
