import rosbag
from cv_bridge import CvBridge
bridge = CvBridge()
from pathlib import Path

class Analyzer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.bag_name = self.name + ".bag"
        self.bag = None

    def open_bag(self):
        try:
            bag = f"UE-HRI/bags/{self.name}.bag"
            assert Path(bag).exists(), f"Cannot find {bag}!"
            self.bag = rosbag.Bag(bag)
        except Exception as e:
            print(f"Unable to read bag {self.name}: {e}")

    def run(self):
        init = 0
        for topic, msg, t in self.bag.read_messages():
            # print(topic)
            if topic == "/naoqi_driver_node/camera/front/image_raw":
                if init == 0:
                    init = t.to_sec()
                store_time = t.to_sec() - init
                img = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
                self.prep.append([t, self.get_num_faces(img)])
                # front_frames.append({"ts": store_time, "image": img.tobytes()})
            if topic == "/naoqi_driver_node/camera/bottom/image_raw":
                if init == 0:
                    init = t.to_sec()
                store_time = t.to_sec() - init
                img = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
                # bottom_frames.append({"ts": store_time, "image": img.tobytes()})
    
    @property
    def duration(self):
        d = self.bag.get_end_time() - self.bag.get_start_time()
        return d

def run():
    a = Analyzer("user106_2017-03-08")
    a.open_bag()

    print()


if __name__=='__main__':
    run()
# def analize(bag_name):
#     print(f"analysing bag {bag_name}")

#     data = STATS_UEHRI(name=bag_name)

#     if data.bag is None:
#         # Download bag
#         dl = GETBAG()
#         dl.get_bag(bag_name + ".bag")
#         print()
#     data.__init__(bag_name)

#     data.store_topics()

#     return Code.SUCCESS


# def empty_space():
#     print("emptying space")
#     dir = "UE-HRI/bags/"
#     filelist = glob.glob(os.path.join(dir, "*.bag"))
#     for f in filelist:
#         os.remove(f)


# def run_stats():
#     # if all_1.csv exists, read from there. Else read from all.csv

#     src = "all.csv"
#     all_df = pd.read_csv(src, header=0)
#     sort_df = all_df.sort_values("name")
#     # print(sort_df.iloc[1,0])
#     t0 = time.time()
#     print(len(sort_df))
#     for n, s in sort_df.iterrows():
#         # print(n)
#         # print(f"{s[0]} {str(s[1]).strip()}")
#         eaf_name = str(s[0]).strip()
#         bag_name = s[0].split(".eaf")[0]
#         init = time.time()
#         ret = analize(bag_name)
#         ret = None
#         sort_df.iloc[n, 1] = ret
#         sort_df.to_csv(store_progress, index=None)
#         print(f"analized {bag_name} in {time.time()-init:.3f} seconds")
#         # print(s)
#         empty_space()
#     duration = time.time() - t0
#     print(f"Analized all bags in {duration:.3f} seconds")
#     print()
