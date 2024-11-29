#!/usr/bin/env python3

import rospy
import message_filters
from std_msgs.msg import String, Float32

# the problem is that these messages are locally build, but in reality they are made by other packages.
try:
    from grace_common_msgs.msg import EngagementValue as Eng4
    from grace_common_msgs.msg import EngValue as Eng5
    # print(EngagementValue)
except Exception as e:
    print(f"Cannot import: {e}")
import csv
import time

class Compare:
    def __init__(self):
        self.eng_4 = None
        self.eng_5 = None
        self.bag = None
        self.progress = None
        self.data_collection_active = False
        self.data = []
        
        rospy.Subscriber("/humans/interactions/engagements", Eng4, self.cb1)
        rospy.Subscriber("/mutual_engagement", Eng5, self.cb2)
        rospy.Subscriber("/bag_name", String, self.cb_bag_name)
        rospy.Subscriber("/bag_progress", Float32, self.cb_progress)
        # Subscribers
        # sub_eng_1 = message_filters.Subscriber("/humans/interactions/engagements", Eng4)
        # sub_eng_2 = message_filters.Subscriber("/mutual_engagement", Eng5)
        self.trigger_sub = rospy.Subscriber("/trigger_topic", String, self.trigger_callback)
        self.timer1 = rospy.Timer(rospy.Duration(0.2), self.main_cb)
        self.timer2 = rospy.Timer(rospy.Duration(4200), self.save_hourly)
        # # Time synchronizer
        # self.ts = message_filters.ApproximateTimeSynchronizer([sub_eng_1, sub_eng_2], queue_size=10, slop=0.1, allow_headerless=True)
        # self.ts.registerCallback(self.callback)
        # rospy.delete_param("/use_sim_time")

        rospy.loginfo("Initialized OK!")
    
    def save_hourly(self, event):
        self.dump_data_to_csv()
        self.data_collection_active = True
        
    def main_cb(self, event):
        if self.data_collection_active:
            timestamp = rospy.Time.now()
            self.data.append([timestamp, self.bag, self.progress, self.eng_4, self.eng_5])
            rospy.loginfo(f"Ts: {timestamp}, bag: {self.bag}, progress: {self.progress}, eng_4: {self.eng_4}, eng_5: {self.eng_5}")
        
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
    
    def cb1(self, msg):
        self.eng_4 = msg.engagement
        # if msg.engagement == self.eng_4:
        #     self.eng_4 = msg.engagement
        #     print(f"eng_4: {self.eng_4}")
        # else:
        #     self.eng_4 = ""
        
    def cb2(self, msg):
        self.eng_5 = msg.engagement
        # if msg.engagement == self.eng_5:
        #     self.eng_5 = msg.engagement
        #     print(f"grace: {self.eng_5}")
        # else:
        #     self.eng_5 = ""

    # def callback(self, eng1, eng2):
    #     print('callback called')
    #     if self.data_collection_active:
    #         # HOW ABOUT THE TIME?
    #         # HOW ABOUT THE ENG1.DATA?
    #         timestamp = time.time()
    #         print(dir(eng2))
    #         self.data.append([timestamp, eng1.data, eng2.data])
    #         rospy.loginfo(f"Collected data - Timestamp: {timestamp}, eng1: {eng1.data}, eng2: {eng2.data}")

    def dump_data_to_csv(self):
        if not self.data:
            rospy.logwarn("No data to write to CSV.")
            return

        filename = f"data/engagement_data_{int(time.time())}.csv"
        with open(filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "name", "progress", "eng_4", "eng_5"])
            writer.writerows(self.data)
        
        rospy.loginfo(f"Data dumped to {filename}")
        self.data.clear()

if __name__ == '__main__':
    rospy.init_node('compare')
    Compare()
    rospy.spin()
