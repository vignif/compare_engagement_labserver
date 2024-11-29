import rosbag
from cv_bridge import CvBridge

bridge = CvBridge()
from pathlib import Path
import cv2 as cv
import numpy as np
from imagehash import average_hash
from PIL import Image
import hashlib
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(self, name: str, options: dict) -> None:
        self.name = name
        self.bag_name = self.name + ".bag"
        self.bag = None
        self.prep = []
        self.frozen = []
        self.options = options

    def open_bag(self):
        try:
            bag = f"UE-HRI/bags/{self.name}.bag"
            assert Path(bag).exists(), f"Cannot find {bag}!"
            self.bag = rosbag.Bag(bag)
            print(f'bag {self.name} opened correctly!')
        except Exception as e:
            print(f"Unable to read bag {self.name}: {e}")

    def run(self):
        init = 0
        old_img = ""
        self.save_t = []
        counter = 0
        for topic, msg, t in self.bag.read_messages():
            # print(topic)
            if topic == "/naoqi_driver_node/camera/front/image_raw":
                if init == 0:
                    init = t.to_sec()
                store_time = t.to_sec() - init

                self.save_t.append([counter, store_time])
                if self.options["count_faces"]:
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)

                    # count number of faces per each frame
                    self.prep.append([store_time, self.get_num_faces(img)])

                    # has video frozen?
                    current = hashlib.sha1(img).hexdigest()

                    # print(store_time)
                    if old_img == current:
                        self.frozen.append([store_time, True])
                    else:
                        self.frozen.append([store_time, False])

                    old_img = hashlib.sha1(img).hexdigest()
                counter += 1

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

    @staticmethod
    def get_num_faces(img):
        detector = cv.FaceDetectorYN.create(
            "face_detection_yunet_2023mar.onnx", "", (640, 480)
        )
        # If input is an image
        tm = cv.TickMeter()
        detector.setInputSize((img.shape[1], img.shape[0]))
        faces = detector.detect(img)
        if faces[1] is not None:
            num_faces = len(faces[1])
            return num_faces
        return 0

    def plot_num_of_faces(self):
        m = min(self.prep)[0]
        faces = np.array(self.prep)
        plt.plot(faces[:, 1], faces[:, 0])
        plt.title(f"Faces in {self.name}")
        plt.savefig("my_plot.png")

    def plot_time(self):
        # plt.plot(self.save_t[0], self.save_t[1])

        t = np.array(self.save_t)
        plt.plot(t[:, 0], t[:, 1])
        # t[:,0] counter
        # t[:,1] seconds
        plt.title(f"Time in {self.name}")
        plt.xlabel("Consecutive Message")
        plt.ylabel("TimeStamp")

        plt.savefig(f"{self.name}_time.png")
        print(f"Plotted time of {self.name}")


def run(filename):
    
    a = Analyzer(filename, options={"count_faces": False})
    a.open_bag()
    a.run()
    # a.plot_num_of_faces()
    a.plot_time()
    print()


if __name__ == "__main__":
    # Import the library
    import argparse
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--name', type=str, required=True)
    # Parse the argument
    args = parser.parse_args()

    run(args.name)
