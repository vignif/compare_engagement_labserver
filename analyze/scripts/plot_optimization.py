import matplotlib.pyplot as plt
import optuna

# Load study results
study = optuna.load_study(study_name='no-name-c4ec09c6-246a-41f8-89ab-298b28926b12', storage='sqlite:///optuna_study.db')

# Get trial numbers and corresponding MSE values
trials = study.trials
trial_numbers = [trial.number for trial in trials]
mse_values = [trial.value for trial in trials]

# Plot the convergence
plt.figure(figsize=(10, 6))
plt.plot(trial_numbers, mse_values, label='Weighted MSE', color='b', lw=2)
plt.axvline(x=study.best_trial.number, color='r', linestyle='--', label='Optimal Trial')

# Highlight the optimal trial
plt.scatter(study.best_trial.number, study.best_value, color='r', s=100, zorder=5)
plt.text(study.best_trial.number, study.best_value, f"Optimal MSE = {study.best_value:.4f}", 
         verticalalignment='bottom', horizontalalignment='right', fontsize=10)

# Add titles and labels
plt.title('Optimization Convergence Plot', fontsize=14)
plt.xlabel('Trial Number', fontsize=12)
plt.ylabel('Weighted MSE', fontsize=12)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
