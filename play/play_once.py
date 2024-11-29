#!/usr/bin/env python3

import rospy
import rosbag
import sys
import time
import signal
from std_msgs.msg import String, Float32
import os

wanted_topics = [
    "/naoqi_driver_node/camera/front/image_raw",
    "/naoqi_driver_node/camera/front/camera_info",
    "/naoqi_driver_node/camera/depth/image_raw",
    "/naoqi_driver_node/camera/depth/camera_info",
    "/tf"
]

def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)

class RosbagReplayer:
    def __init__(self, bag_file):
        self.bag_file = bag_file
        self.stop_replay = False
        self.pub = rospy.Publisher("/bag_name", String, queue_size=10, latch=True)
        self.pub_progress = rospy.Publisher("/bag_progress", Float32, queue_size=10, latch=True)

    def _signal_handler(self, sig, frame):
        print("\nCtrl+C detected! Stopping replay...")
        self.stop_replay = True

    def replay_bag(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        with rosbag.Bag(self.bag_file) as bag:
            start_time = bag.get_start_time()
            end_time = bag.get_end_time()
            total_duration = end_time - start_time
            print(f"Rosbag named: {self.bag_file}")
            self.pub.publish(String(self.bag_file))
            print(f"Total duration of the bag: {total_duration:.2f} seconds")
            print(f"Replaying topics: {', '.join(wanted_topics)}")

            start_time_realtime = time.time()
            last_display_time = 0

            for topic, msg, t in bag.read_messages(topics=wanted_topics):
                if self.stop_replay:
                    break

                pub = rospy.Publisher(topic, type(msg), queue_size=10)
                pub.publish(msg)

                current_time_realtime = time.time()
                elapsed_time_realtime = current_time_realtime - start_time_realtime
                elapsed_time_bag = t.to_sec() - start_time
                progress = (elapsed_time_bag / total_duration) * 100

                if current_time_realtime - last_display_time >= 1:
                    last_display_time = current_time_realtime
                    p = float(elapsed_time_bag / total_duration)
                    self.pub_progress.publish(Float32(p))

                sleep_time = (t.to_sec() - start_time) - elapsed_time_realtime
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print("Replay loop completed for bag: ", self.bag_file)

if __name__ == "__main__":
    debug = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ns = dir_path + "/bags/"
    if debug:
        # take as threshold 2 STD
        bag_files = [
                ns + "user1_2017-03-03.bag",
                ns + "user108_2017-03-15.bag",
                ns + "user14_2017-06-14.bag",
                ns + "user16_2017-01-23.bag",
                ns + "user184_2017-02-03.bag",
                ns + "user191_2017-03-16.bag",
                ns + "user201_2017-02-03.bag",
                ns + "user218_2017-02-09.bag",
                ns + "user23_2017-01-20.bag",
                ns + "user230_2017-01-25.bag",
                ns + "user279_2017-04-26.bag",
                # ns + "user28_2017-01-26.bag", # INVALID ROSBAG
                ns + "user28_2017-01-31.bag",
                ns + "user315_2017-02-10.bag",
                ns + "user36_2017-03-13.bag",
                ns + "user4_2017-02-17.bag",
                # ns + "user41_2017-02-07.bag", # INVALID ROSBAG
                ns + "user555_2017-04-14.bag",
                ns + "user60_2017-02-20.bag",
                ns + "user66_2017-05-12.bag",
                ns + "user8_2017-01-31.bag",
                ns + "user8_2017-03-23.bag",
                ns + "user95_2017-02-24.bag",
                ns + "user104_2017-06-20.bag",
                ns + "user2_2017-01-19.bag",
                ns + "user215_2017-02-03.bag",
                ns + "user23_2017-03-10.bag",
                # ns + "user307_2017-04-13.bag", # INVALID ROSBAG
                # ns + "user334_2017-04-13.bag", # INVALID ROSBAG
                ns + "user350_2017-04-13.bag",
                ns + "user53_2017-01-31.bag",
                ns + "user68_2017-01-26.bag"
        ]
    else:
        if len(sys.argv) < 2:
            print("Usage: play_rosbag.py <bag_file1> <bag_file2> ... <bag_fileN>")
            sys.exit(1)

        bag_files = sys.argv[1:]

    rospy.init_node('rosbag_replay', anonymous=False)

    rospy.loginfo("About to start playing the rosbags")
    time.sleep(10)
    final_csv_dump = rospy.Publisher("/trigger_topic", String, queue_size=10, latch=True)

    for bag_file in bag_files:
        try:
            print(f"About to play: {bag_file}")
            player = RosbagReplayer(bag_file)
            player.replay_bag()
        except rospy.ROSInterruptException:
            pass
        except Exception as e:
            signal.signal(signal.SIGINT, sigterm_handler)
            signal.signal(signal.SIGTERM, sigterm_handler)
            raise Exception(f"Sorry {e}")

    print("Completed sequence of bags. Exiting.")
    final_csv_dump.publish(String("stop"))
    time.sleep(10)    
    rospy.signal_shutdown("Play Finished")
