import matplotlib.pyplot as plt
import optuna

def load_study_from_disk(study_name="main", storage="sqlite:///k_fold.db"):
    """
    Load an Optuna study from disk.

    Args:
        study_name (str): The name of the study to load.
        storage (str): The path to the SQLite database.

    Returns:
        optuna.Study: The loaded study.
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study

# Load the study
study = load_study_from_disk()

# Optionally, print best parameters and value for verification
print("Best parameters:", study.best_params)
print("Best cross-validated MSE:", study.best_value)


def plot_cross_validation_error(study):
    """
    Plot the cross-validation error by trial number.
    
    Args:
        study (optuna.Study): The study containing the trial results.
    """
    # Extract trial numbers and their corresponding cross-validated MSE values
    trial_numbers = [trial.number for trial in study.trials]
    trial_errors = [trial.value for trial in study.trials]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(trial_numbers, trial_errors, marker='o', linestyle='-', color='b', label='Cross-validated MSE')
    plt.xlabel('Trial Number')
    plt.ylabel('Cross-Validated MSE')
    plt.title('Cross-Validation Error by Trial')
    plt.legend()
    plt.grid(True)
    plt.show()

# Assuming you have an Optuna study created as `study`
plot_cross_validation_error(study)
