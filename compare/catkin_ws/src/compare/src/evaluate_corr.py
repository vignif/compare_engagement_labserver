#!/usr/bin/env python3

import rospy
import message_filters
from std_msgs.msg import String, Float32
import csv
import time
import os
import random

# the problem is that these messages are locally built, but in reality they are made by other packages.
try:
    from grace_common_msgs.msg import EngagementValue as Eng4
    from grace_common_msgs.msg import EngValue as Eng5
    from grace_common_msgs.srv import GetEngParams
except Exception as e:
    rospy.logerr(f"Cannot import: {e}")

class Evaluate:
    def __init__(self):
        # optimals
        self.eps_val = 2.3266715076954303
        self.prox_w_val = 0.31003009283282934
        self.gaze_w_val = 0.6894529665209421
        # 
        # self.eps_val = 0.5
        # self.prox_w_val = 0.7
        # self.gaze_w_val = 0.3
        self.eng_1 = 0.0
        self.eng_4 = 0.0
        self.eng_grace = 0.0 # grace value
        self.bag = None
        self.progress = None
        self.data_collection_active = True
        self.data = []
        
        rospy.Subscriber("/humans/interactions/engagements", Eng4, self.cb_eng4) # tamlin eng_4
        rospy.Subscriber("/engagement_detector/value", Float32, self.cb_eng1) # eng_1
        rospy.Subscriber("/bag_name", String, self.cb_bag_name)
        rospy.Subscriber("/bag_progress", Float32, self.cb_progress)
        rospy.Subscriber("/trigger_topic", String, self.trigger_callback)
        
        try:
            rospy.wait_for_service("get_engagement_params")
            self.compute_engagement = rospy.ServiceProxy("get_engagement_params", GetEngParams)
        except Exception as e:
            rospy.logerr(f"{e}")
        
        self.timer1 = rospy.Timer(rospy.Duration(0.2), self.main_cb)
        self.timer2 = rospy.Timer(rospy.Duration(1800), self.save_hourly) # every 30min
        self.timer3 = rospy.Timer(rospy.Duration(0.01), self.get_engagement_grace)
        rospy.loginfo("Initialized OK!")
    
    def get_engagement_grace(self, event):
        try:
            # OPTIMAL PARAMETERS FOUND
            self.e = self.compute_engagement(self.eps_val, self.prox_w_val, self.gaze_w_val)
            self.eng_grace = self.e.engagement
            # rospy.loginfo(f"Params set to {self.eps_val:.3f}, {self.prox_w_val:.3f}, {self.gaze_w_val:.3f}")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
        
    def save_hourly(self, event):
        self.dump_data_to_csv()
        self.data_collection_active = True
        
    def main_cb(self, event):
        if self.data_collection_active:
            timestamp = rospy.Time.now().to_sec()
            self.data.append([
                timestamp, self.bag, self.progress,
                self.eng_4, self.eng_1, self.eng_grace,
                self.eps_val, self.prox_w_val, self.gaze_w_val
            ])
            rospy.loginfo(f"Ts: {timestamp}, bag: {self.bag}, progress: {self.progress}, eng_4: {self.eng_4}, eng_1: {self.eng_1}, grace_eng: {self.eng_grace}, eps: {self.eps_val}, prox_weight: {self.prox_w_val}, gaze_weight: {self.gaze_w_val}")
        
    def trigger_callback(self, msg):
        if msg.data == "start":
            rospy.loginfo("Data collection started.")
            self.data_collection_active = True
        elif msg.data == "stop":
            rospy.loginfo("Data collection stopped. Dumping data to CSV.")
            self.data_collection_active = False
            self.dump_data_to_csv()
    
    def cb_bag_name(self, msg):
        self.bag = msg.data
    
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
        filename = f"data/eval_{int(time.time())}.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["timestamp", "name", "progress", "eng_4", "eng_1", "grace_eng", "prox_epsilon", "prox_weight", "gaze_weight"])
            writer.writerows(self.data)
        
        rospy.loginfo(f"Data dumped to {filename}")
        self.data.clear()

if __name__ == '__main__':
    rospy.init_node('Evaluate')
    Evaluate()
    rospy.spin()
