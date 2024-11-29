import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_learning_steps(learning_steps_df):
    """Plot the learning steps."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=learning_steps_df, x='prox_epsilon', y='average_mse', hue='prox_weight', style='gaze_weight', markers=True)
    plt.title('Learning Steps: Average MSE by Prox Epsilon, Prox Weight, and Gaze Weight')
    plt.xlabel('Prox Epsilon')
    plt.ylabel('Average MSE')
    plt.legend(title='Prox Weight & Gaze Weight', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig('learning_steps_plot.png')
    plt.show()

def plot_model_performance(val_mse, test_mse):
    """Plot validation and test MSE."""
    plt.figure(figsize=(6, 4))
    plt.bar(['Validation MSE', 'Test MSE'], [val_mse, test_mse], color=['blue', 'orange'])
    plt.title('Model Performance')
    plt.ylabel('Mean Squared Error')
    plt.ylim(0, max(val_mse, test_mse) + 1)  # Adjust y-axis limit
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('model_performance_plot.png')
    plt.show()
