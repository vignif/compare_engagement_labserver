import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import json
from sklearn.model_selection import KFold
from data_prep import load_data  # Loading the complete data, without splitting in advance

class FindParams:
    """
    A class to load a dataset and optimize the parameters prox_epsilon, prox_weight,
    and gaze_weight using Optuna to minimize the weighted mean squared error with cross-validation.
    """

    def __init__(self, df, n_splits=20):
        """
        Initializes the FindParams class.

        Args:
            df (pd.DataFrame): The complete dataset.
            n_splits (int): Number of cross-validation splits.
        """
        self.dataframe = df
        self.n_splits = n_splits

    def _error_function(self, trial):
        """
        Calculates the average weighted mean squared error (MSE) over cross-validation folds
        for given parameters suggested by Optuna.

        Args:
            trial (optuna.trial.Trial): A trial object which suggests values for parameters.

        Returns:
            float: The cross-validated weighted mean squared error.
        """
        # Suggest parameter values within their respective ranges
        prox_epsilon = trial.suggest_uniform('prox_epsilon', self.dataframe['prox_epsilon'].min(), self.dataframe['prox_epsilon'].max())
        prox_weight = trial.suggest_uniform('prox_weight', 0, 1)
        gaze_weight = trial.suggest_uniform('gaze_weight', 0, 1)

        # Set up KFold cross-validation
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold_errors = []

        for train_idx, val_idx in kf.split(self.dataframe):
            train_df = self.dataframe.iloc[train_idx]
            val_df = self.dataframe.iloc[val_idx]

            # Calculate distances and weights for the validation fold
            distances = np.abs(val_df['prox_epsilon'] - prox_epsilon) + \
                        np.abs(val_df['prox_weight'] - prox_weight) + \
                        np.abs(val_df['gaze_weight'] - gaze_weight)
            weights = 1 / (distances + 1e-6)
            
            # Compute the weighted mean squared error for the validation fold
            weighted_mse = np.sum(weights * ((val_df['eng_1'] - val_df['grace_eng']) ** 2)) / np.sum(weights)
            fold_errors.append(weighted_mse)

        # Return the average error across all folds
        return np.mean(fold_errors)

    def optimize_parameters(self, n_trials=3000, patience=1000, min_delta=1e-5, seed=42, results_file='results1.json'):
        """
        Optimize prox_epsilon, prox_weight, and gaze_weight to minimize the weighted MSE
        with cross-validation and early stopping.

        Args:
            n_trials (int): Maximum number of trials for optimization.
            patience (int): Number of trials to wait before stopping if no improvement is found.
            min_delta (float): Minimum difference in the MSE to be considered an improvement.
            seed (int): Random seed for reproducibility.
            results_file (str): Path to save the results.

        Returns:
            dict: The best parameter values found.
        """
        def early_stopping_callback(study, trial):
            if len(study.trials) > patience:
                best_trials = study.trials[-patience:]
                if all(np.abs(trial.value - best_trials[0].value) < min_delta for trial in best_trials):
                    study.stop()

        sampler = TPESampler(seed=seed)
        storage = "sqlite:///k_fold.db"
        # study = optuna.create_study(direction='minimize', storage=storage, load_if_exists=True, sampler=sampler)
        study = optuna.create_study(
            study_name='main',
            direction='minimize',
            storage=storage,
            load_if_exists=True,
            sampler=sampler
        )
        
        study.optimize(self._error_function, n_trials=n_trials, callbacks=[early_stopping_callback])

        best_params = study.best_params
        best_mse = study.best_value

        # Output the results
        print("Optimal parameters found:")
        print(f"prox_epsilon: {best_params['prox_epsilon']}")
        print(f"prox_weight: {best_params['prox_weight']}")
        print(f"gaze_weight: {best_params['gaze_weight']}")
        print(f"Overall weighted MSE at optimal parameters: {best_mse}")

        self._save_results(best_params, best_mse, study.trials, results_file)
        return best_params

    def _save_results(self, best_params, best_mse, trials, results_file):
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
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {results_file}")

def main():
    """
    Main function to run the parameter optimization.
    """
    # Load the entire dataset without splitting
    df = load_data('dataset.csv')
    fp = FindParams(df)
    optimal_params = fp.optimize_parameters()

if __name__ == "__main__":
    main()
