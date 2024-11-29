import json
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# Load the JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Extract relevant variables for plotting
def extract_variables(data):
    thresholds = []
    uses = []  # Percentage of dataset used
    prox_eps_means = []
    
    for entry in data:
        thresholds.append(entry['threshold'])
        prox_eps_means.append(entry['prox_epsilon']['mean'])
        
        uses.append(entry['used_data'])

    
    return np.array(thresholds), np.array(uses), np.array(prox_eps_means)

# Create a 3D plot
def create_3d_plot(thresholds, uses, prox_eps_means):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(thresholds, uses, prox_eps_means, c='b', marker='o')
    
    # Labels and title
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Use (%)')
    ax.set_zlabel('Prox Eps Mean')
    ax.set_title('3D Plot of Threshold, Use, and Prox Eps Mean')

    plt.show()

if __name__ == "__main__":
    # Replace with the path to your aggregated results JSON file
    file_path = 'aggregated_results1.json'
    
    # Load data from the JSON file
    data = load_data(file_path)
    
    # Extract the necessary variables
    thresholds, uses, prox_eps_means = extract_variables(data)
    
    # Create the 3D plot
    create_3d_plot(thresholds, uses, prox_eps_means)
