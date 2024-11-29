#!/usr/bin/env python3

import rospy
import message_filters
from std_msgs.msg import String, Float32
import csv
import time
import os
import random

try:
    from grace_common_msgs.msg import EngagementValue as Eng4
    from grace_common_msgs.msg import EngValue as Eng5
    from grace_common_msgs.srv import GetEngParams
except ImportError as e:
    rospy.logerr(f"Failed to import required messages or services: {e}")
    raise

class CreateDB:
    def __init__(self):
        self.eps_val = 0.5
        self.prox_w_val = 0.5
        self.gaze_w_val = 0.5
        self.eng_1 = 0.0
        self.eng_4 = 0.0
        self.eng_grace = 0.0 # grace value
        self.eng_grace_gaze = 0.0
        self.eng_grace_prox = 0.0
        self.bag = None
        self.progress = None
        self.data_collection_active = False
        self.data = []
        self.compute_engagement = None  # Assign None if the service fails
        
        rospy.Subscriber("/humans/interactions/engagements", Eng4, self.cb_eng4) # tamlin eng_4
        rospy.Subscriber("/engagement_detector/value", Float32, self.cb_eng1) # eng_1
        rospy.Subscriber("/bag_name", String, self.cb_bag_name)
        rospy.Subscriber("/bag_progress", Float32, self.cb_progress)
        rospy.Subscriber("/trigger_topic", String, self.trigger_callback)
               
        self.timer1 = rospy.Timer(rospy.Duration(0.2), self.main_cb)
        self.timer2 = rospy.Timer(rospy.Duration(4200), self.save_hourly)
        self.timer3 = rospy.Timer(rospy.Duration(1800), self.shuffle_params)
        self.timer4 = rospy.Timer(rospy.Duration(0.1), self.get_eng)

        try:
            rospy.wait_for_service("get_engagement_params", timeout=60)
            self.compute_engagement = rospy.ServiceProxy("get_engagement_params", GetEngParams)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
        
        rospy.loginfo("Initialized OK!")

    def get_eng(self, event):
        if self.compute_engagement is None:
            rospy.logwarn("Compute engagement service is not available.")
            return  # Early exit if the service is not available
        
        valid_engagement = False  # Flag to indicate valid engagement found

        while not valid_engagement:
            # Compute engagement
            try:
                e = self.compute_engagement(self.eps_val, self.prox_w_val, self.gaze_w_val)
                
                # Log the current parameters and engagement value
                rospy.loginfo(f"Params {self.eps_val:.3f}, {self.prox_w_val:.3f}, {self.gaze_w_val:.3f}. eng: {e.engagement:.3f}")
                
                # Check if engagement is valid
                if e.success:
                    valid_engagement = True  # Exit loop if a valid engagement is found
                    self.eng_grace = e.engagement  # Store the valid engagement value
                    self.eng_grace_gaze = e.engagement_gaze
                    self.eng_grace_prox = e.engagement_prox
                    rospy.loginfo("Valid engagement found and stored.")
                else:
                    rospy.logwarn("Invalid engagement computed, retrying...")
            except rospy.ServiceException as e:
                rospy.logerr(f"Error during service call: {e}")
                break  # Exit if the service fails during the call

    def shuffle_params(self, event):
        self.eps_val = random.uniform(0, 4)
        self.prox_w_val = random.uniform(0, 1)
        self.gaze_w_val = 1 - self.prox_w_val
        rospy.loginfo(f"Parameters shuffled: eps={self.eps_val:.3f}, prox_w={self.prox_w_val:.3f}, gaze_w={self.gaze_w_val:.3f}")

    def save_hourly(self, event):
        self.dump_data_to_csv()

    def main_cb(self, event):
        if self.data_collection_active:
            timestamp = rospy.Time.now().to_sec()
            self.data.append([
                timestamp, self.bag, self.progress,
                self.eng_4, self.eng_1, self.eng_grace,
                self.eps_val, self.prox_w_val, self.gaze_w_val,
                self.eng_grace_prox, self.eng_grace_gaze
            ])
            rospy.loginfo(f"Data recorded: ts={timestamp}, bag={self.bag}, eng_grace={self.eng_grace:.3f}")

    def trigger_callback(self, msg):
        if msg.data == "start":
            self.data_collection_active = True
            rospy.loginfo("Data collection started.")
        elif msg.data == "stop":
            self.data_collection_active = False
            rospy.loginfo("Data collection stopped. Dumping data.")
            self.dump_data_to_csv()

    def cb_bag_name(self, msg):
        self.bag = msg.data
        rospy.loginfo(f"Bag name updated: {self.bag}")

    def cb_progress(self, msg):
        self.progress = msg.data

    def cb_eng4(self, msg):
        self.eng_4 = msg.engagement

    def cb_eng1(self, msg):
        self.eng_1 = msg.data

    def dump_data_to_csv(self):
        if not self.data:
            rospy.logwarn("No data to write to CSV.")
            return
        
        os.makedirs("data", exist_ok=True)
        filename = f"data/engagement_data_{int(time.time())}.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([
                "timestamp", "name", "progress", "eng_4", "eng_1",
                "grace_eng", "prox_epsilon", "prox_weight", "gaze_weight",
                "grace_prox", "grace_gaze"
            ])
            writer.writerows(self.data)
        
        rospy.loginfo(f"Data saved to {filename}")
        self.data.clear()

if __name__ == '__main__':
    rospy.init_node('CreateDB', anonymous=True)
    try:
        CreateDB()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")
