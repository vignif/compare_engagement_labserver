import pandas as pd
from sklearn.model_selection import train_test_split
from load_data import load_data
from optimize_parameters import optimize_parameters
from train_model import train_and_evaluate
from plot_results import plot_learning_steps, plot_model_performance

def main():
    # Load the dataset
    df = load_data('dataset.csv')  # Replace 'dataset.csv' with your actual dataset file path
    
    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Further split the training set into training and validation sets
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Optimize parameters based on training data
    best_params, best_mse, learning_steps = optimize_parameters(train_df)

    # Train and evaluate the model
    val_mse, test_mse = train_and_evaluate(train_df, val_df, test_df, best_params)

    # Convert learning steps to DataFrame for easy manipulation
    learning_steps_df = pd.DataFrame(learning_steps, columns=['prox_epsilon', 'prox_weight', 'gaze_weight', 'average_mse'])

    # Save learning steps to a CSV file for later use
    learning_steps_df.to_csv('learning_steps.csv', index=False)

    # Plot learning steps and model performance
    plot_learning_steps(learning_steps_df)
    plot_model_performance(val_mse, test_mse)

    print("Optimal parameters found:")
    print(f"prox_epsilon: {best_params[0]}")
    print(f"prox_weight: {best_params[1]}")
    print(f"gaze_weight: {best_params[2]}")
    print(f"Average MSE from cross-validation: {best_mse}")
    print(f"Validation MSE: {val_mse}")
    print(f"Test MSE: {test_mse}")

if __name__ == "__main__":
    main()
