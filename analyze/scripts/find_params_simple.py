import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import argparse
from pathlib import Path

class FindParams:
    """
    A class to load a dataset and optimize the parameters prox_epsilon, prox_weight,
    and gaze_weight using Optuna to minimize the weighted mean squared error.
    """

    def __init__(self, file_path):
        """
        Initializes the FindParams class.

        Args:
            file_path (str or Path): Path to the CSV dataset file.
        """
        self.file_path = file_path
        self.dataframe = self._load_dataframe()

    def _load_dataframe(self):
        """
        Loads the dataset from the CSV file.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            ValueError: If there's an error loading the CSV file.
        """
        try:
            return pd.read_csv(self.file_path, sep=';')
        except Exception as e:
            raise ValueError(f"Error loading DataFrame: {e}")

    def _error_function(self, trial):
        """
        Calculates the weighted mean squared error (MSE) for given parameters
        suggested by Optuna.

        Args:
            trial (optuna.trial.Trial): A trial object which suggests values for parameters.

        Returns:
            float: The weighted mean squared error.
        """
        # Suggest parameter values within their respective ranges
        prox_epsilon = trial.suggest_uniform('prox_epsilon', self.dataframe['prox_epsilon'].min(), self.dataframe['prox_epsilon'].max())
        prox_weight = trial.suggest_uniform('prox_weight', 0, 1)
        gaze_weight = trial.suggest_uniform('gaze_weight', 0, 1)

        # Calculate distances for each row between current values and the suggested parameters
        distances = np.abs(self.dataframe['prox_epsilon'] - prox_epsilon) + \
                    np.abs(self.dataframe['prox_weight'] - prox_weight) + \
                    np.abs(self.dataframe['gaze_weight'] - gaze_weight)
        
        # Avoid division by zero by adding a small value (1e-6)
        weights = 1 / (distances + 1e-6)
        
        # Compute the weighted mean squared error
        weighted_mse = np.sum(weights * ((self.dataframe['eng_1'] - self.dataframe['grace_eng']) ** 2)) / np.sum(weights)
        
        return weighted_mse

    def optimize_parameters(self, n_trials=2000, patience=1000, min_delta=1e-4, seed=42):
        """
        Optimize prox_epsilon, prox_weight, and gaze_weight to minimize the weighted MSE
        with early stopping based on convergence and a fixed seed for reproducibility.

        Args:
            n_trials (int): Maximum number of trials for optimization.
            patience (int): Number of trials to wait before stopping if no improvement is found.
            min_delta (float): Minimum difference in the MSE to be considered an improvement.
            seed (int): Random seed for reproducibility.
        
        Returns:
            dict: The best parameter values found.
        """
        # Callback to stop if no improvement
        def early_stopping_callback(study, trial):
            if len(study.trials) > patience:
                best_trials = study.trials[-patience:]
                if all(np.abs(trial.value - best_trials[0].value) < min_delta for trial in best_trials):
                    study.stop()

        # Create the study with a fixed seed for reproducibility
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(self._error_function, n_trials=n_trials, callbacks=[early_stopping_callback])

        best_params = study.best_params
        best_mse = study.best_value

        # Output the results
        print("Optimal parameters found:")
        print(f"prox_epsilon: {best_params['prox_epsilon']}")
        print(f"prox_weight: {best_params['prox_weight']}")
        print(f"gaze_weight: {best_params['gaze_weight']}")
        print(f"Overall weighted MSE at optimal parameters: {best_mse}")

        return best_params

def main():
    """
    Main function to parse command line arguments and run the parameter optimization.
    """
    parser = argparse.ArgumentParser(description='Optimize parameters to minimize weighted MSE.')
    parser.add_argument('--file_path', type=str, help='Path to the CSV file', default="dataset.csv")

    args = parser.parse_args()
    file_path = Path(args.file_path)

    # Initialize and run parameter optimization
    fp = FindParams(file_path)

    try:
        optimal_params = fp.optimize_parameters()
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
