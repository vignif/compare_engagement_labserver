import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class GRACEOptimizer:
    def __init__(self, dataset_path):
        """
        Initialize optimizer with dataset
        
        Args:
            dataset_path (str): Path to CSV file
        """
        # Read CSV with semicolon separator
        self.data = pd.read_csv(dataset_path, sep=';')
        
        # Prepare target metric
        self.y_e2e = self.data['eng_1']
    
    def grace_model(self, params, data):
        """
        GRACE engagement calculation
        
        Args:
            params (tuple): Optimization parameters 
                (proximity_epsilon, proximity_weight, gaze_weight)
            data (pd.DataFrame): Input data
        
        Returns:
            np.array: Engagement scores from GRACE model
        """
        proximity_epsilon, proximity_weight, gaze_weight = params
        
        # Ensure weights sum to 1.0
        assert np.isclose(proximity_weight + gaze_weight, 1.0), "Weights must sum to 1.0"
        
        # Proximity calculation (binary threshold)
        proximity_score = (data['prox_epsilon'] > proximity_epsilon).astype(float)
        
        # Gaze weight component (using gaze_weight)
        gaze_component = data['gaze_weight'] * gaze_weight
        
        # Engagement calculation
        engagement = (
            proximity_weight * proximity_score + 
            gaze_component
        )
        
        return engagement
    
    def objective_function(self, params):
        """
        Objective function to minimize difference between GRACE and E2E metrics
        
        Args:
            params (tuple): Optimization parameters 
        
        Returns:
            float: Mean squared error between GRACE and E2E metrics
        """
        # Unpack the parameters ensuring last two sum to 1.0
        params = list(params)
        params[2] = 1.0 - params[1]  # Enforce weight constraint
        
        # Compute GRACE engagement
        grace_scores = self.grace_model(params, self.data)
        
        # Return mean squared error
        return mean_squared_error(self.y_e2e, grace_scores)
    
    def optimize_parameters(self):
        """
        Perform parameter optimization using differential evolution
        
        Returns:
            dict: Optimization results
        """
        # Parameter bounds
        # [proximity_epsilon, proximity_weight, (implicit gaze_weight)]
        bounds = [
            (0, 1),    # proximity_epsilon
            (0, 1),    # proximity_weight
        ]
        
        # Differential Evolution optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            strategy='best1bin',
            popsize=15,
            maxiter=100
        )
        
        # Final parameters (with enforced weight constraint)
        optimized_params = list(result.x)
        optimized_params.append(1.0 - optimized_params[1])
        
        # Compute optimized GRACE scores
        grace_scores = self.grace_model(optimized_params, self.data)
        
        # Compute performance metrics
        return {
            'proximity_epsilon': optimized_params[0],
            'proximity_weight': optimized_params[1],
            'gaze_weight': optimized_params[2],
            'mse': mean_squared_error(self.y_e2e, grace_scores),
            'r2_score': r2_score(self.y_e2e, grace_scores)
        }
    
    def visualize_results(self, results):
        """
        Generate comprehensive visualization of optimization results
        
        Args:
            results (dict): Optimization results
        """
        # Compute optimized GRACE scores
        grace_scores = self.grace_model(
            [results['proximity_epsilon'], 
             results['proximity_weight'], 
             results['gaze_weight']], 
            self.data
        )
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(self.y_e2e, grace_scores, alpha=0.5)
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Alignment')
        ax1.set_xlabel('E2E Engagement (Eng_1)')
        ax1.set_ylabel('GRACE Engagement')
        ax1.set_title('Engagement Metric Comparison')
        ax1.legend()
        
        # Residual plot
        residuals = self.y_e2e - grace_scores
        ax2.scatter(grace_scores, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted GRACE Engagement')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        
        plt.tight_layout()
        plt.show()
    
    def generate_scientific_report(self, results):
        """
        Generate a structured report for scientific paper
        
        Args:
            results (dict): Optimization results
        
        Returns:
            str: Formatted report
        """
        report = f"""
GRACE Engagement Model Optimization Results

Optimized Parameters:
- Proximity Epsilon: {results['proximity_epsilon']:.4f}
- Proximity Weight: {results['proximity_weight']:.4f}
- Gaze Weight: {results['gaze_weight']:.4f}

Performance Metrics:
- Mean Squared Error: {results['mse']:.4f}
- R² Score: {results['r2_score']:.4f}

Methodology:
The GRACE engagement model was optimized using differential evolution 
to minimize the discrepancy between the model's engagement scores and 
the E2E engagement metric (Eng_1). The optimization process adjusted 
the proximity epsilon and weights while maintaining the constraint 
that proximity and gaze weights sum to 1.0.

Key Insights:
- Proximity Epsilon determines the threshold for binary proximity scoring
- Proximity Weight and Gaze Weight distribute the engagement calculation
- The optimization seeks to align GRACE model with E2E metric
"""
        return report

# Example usage
def main():
    # Replace with your actual dataset path
    optimizer = GRACEOptimizer('path/to/your/dataset.csv')
    
    # Optimize parameters
    results = optimizer.optimize_parameters()
    
    # Print optimization results
    print(optimizer.generate_scientific_report(results))
    
    # Visualize results
    optimizer.visualize_results(results)

if __name__ == "__main__":
    main()