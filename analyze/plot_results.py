import optuna
import matplotlib.pyplot as plt

def plot_optimization_progress(study_name="optuna_study", storage="sqlite:///optuna_study.db", output_file="optimization_plot.png"):
    """
    Plots the optimization progress and parameter values across trials.

    Args:
        study_name (str): The name of the Optuna study.
        storage (str): The storage location for the Optuna study.
        output_file (str): The file name to save the plot.
    """
    # Load the study
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Extract trial data
    trials = study.trials
    trial_numbers = [trial.number for trial in trials]
    trial_values = [trial.value for trial in trials]

    # Plot the optimization progress
    plt.figure(figsize=(12, 6))
    plt.plot(trial_numbers, trial_values, marker='o', linestyle='-', label='Weighted MSE')
    plt.xlabel("Trial Number", fontsize=14)
    plt.ylabel("Weighted MSE", fontsize=14)
    plt.title("Optimization Progress", fontsize=16)
    plt.grid(alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file)
    plt.show()

def plot_parameter_distributions(study_name="optuna_study", storage="sqlite:///optuna_study.db", output_file="parameter_distributions.png"):
    """
    Plots the distributions of parameters across trials.

    Args:
        study_name (str): The name of the Optuna study.
        storage (str): The storage location for the Optuna study.
        output_file (str): The file name to save the plot.
    """
    # Load the study
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Extract parameter values for each trial
    trials = study.trials
    prox_epsilons = [trial.params['prox_epsilon'] for trial in trials]
    prox_weights = [trial.params['prox_weight'] for trial in trials]
    gaze_weights = [trial.params['gaze_weight'] for trial in trials]

    # Create subplots
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(prox_epsilons, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("prox_epsilon", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Distribution of prox_epsilon", fontsize=14)

    plt.subplot(1, 3, 2)
    plt.hist(prox_weights, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel("prox_weight", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Distribution of prox_weight", fontsize=14)

    plt.subplot(1, 3, 3)
    plt.hist(gaze_weights, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel("gaze_weight", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Distribution of gaze_weight", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def main():
    """
    Main function to generate optimization plots.
    """
    plot_optimization_progress()
    plot_parameter_distributions()

if __name__ == "__main__":
    main()
