#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
import numpy as np
from scipy.optimize import minimize
from grace_common_msgs.msg import EngagementValue as Eng4
from grace_common_msgs.msg import EngValue as Eng5

class MetricOptimizer:
    def __init__(self):
        self.eng4_value = None
        self.grace_value = None
        self.dataset = []

        # Initial guess for parameters
        self.theta = np.array([0.1, 0.5])  # theta[0] is prox_epsilon, theta[1] is prox_weight (w)
        self.previous_theta = np.copy(self.theta)
        self.convergence_threshold = 1e-9
        self.max_iterations = 1e10
        self.iteration_count = 0

        rospy.init_node('metric_optimizer', anonymous=True)

        # Subscribers for the metric outputs
        rospy.Subscriber('/humans/interactions/engagements', Eng4, self.eng4_callback)
        rospy.Subscriber('/mutual_engagement', Eng5, self.grace_callback)

        # Publishers for the metric parameters
        self.prox_epsilon_pub = rospy.Publisher('/prox_epsilon', Float32, queue_size=10)
        self.prox_weight_pub = rospy.Publisher('/prox_weight', Float32, queue_size=10)
        self.gaze_weight_pub = rospy.Publisher('/gaze_weight', Float32, queue_size=10)

        self.rate = rospy.Rate(10)  # 10 Hz

    def eng4_callback(self, msg):
        self.eng4_value = msg.engagement
        rospy.loginfo_throttle(10, f"received eng_4: {self.eng4_value}")

    def grace_callback(self, msg):
        self.grace_value = msg.engagement
        rospy.loginfo_throttle(10, f"received eng_5: {self.grace_value}")

    def loss_function(self, theta):
        if self.eng4_value is None or self.grace_value is None:
            return float('inf')
        
        prox_epsilon = theta[0]
        prox_weight = theta[1]
        gaze_weight = 1 - prox_weight

        # Publish the parameters to their topics
        self.prox_epsilon_pub.publish(prox_epsilon)
        self.prox_weight_pub.publish(prox_weight)
        self.gaze_weight_pub.publish(gaze_weight)

        # Allow time for the parameter effects to take place
        # rospy.sleep(1)  # Adjust if necessary

        # Collect data points
        self.dataset.append((self.eng4_value, self.grace_value))

        # Calculate loss
        loss = 0
        for eng4_value, grace_value in self.dataset:
            loss += (eng4_value - grace_value) ** 2
        return loss / len(self.dataset)

    def optimize_parameters(self):
        bounds = [(0, None), (0, 1)]  # (prox_epsilon > 0, 0 <= prox_weight <= 1)
        result = minimize(self.loss_function, self.theta, method='L-BFGS-B', bounds=bounds)
        return result.x

    def run(self):
        while not rospy.is_shutdown():
            if self.eng4_value is not None and self.grace_value is not None:
                optimal_theta = self.optimize_parameters()
                self.theta = optimal_theta
                self.iteration_count += 1

                rospy.loginfo("Optimal Parameters: %s", optimal_theta)
                
                # Check for convergence
                if np.all(np.abs(self.theta - self.previous_theta) < self.convergence_threshold):
                    rospy.loginfo("Convergence achieved.")
                    break
                if self.iteration_count >= self.max_iterations:
                    rospy.loginfo("Maximum iterations reached.")
                    break
                
                self.previous_theta = np.copy(self.theta)

                prox_epsilon = optimal_theta[0]
                prox_weight = optimal_theta[1]
                gaze_weight = 1 - prox_weight

                # Publish the optimal parameters
                self.prox_epsilon_pub.publish(prox_epsilon)
                self.prox_weight_pub.publish(prox_weight)
                self.gaze_weight_pub.publish(gaze_weight)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        optimizer = MetricOptimizer()
        optimizer.run()
    except rospy.ROSInterruptException:
        pass
