from pathlib import Path
import rosbag
import pympi
import pickle
from cv_bridge import CvBridge
import dill
import os
import numpy as np
import cv2 as cv
import pandas as pd
bridge = CvBridge()



class UEHRI_DATA(dict):
    """
    Convert a bag file and its annotations to a pickle file
    Read the pickle file

    Data structure:
        'front':[timestamp, image]
        'bottom':[timestamp, image]
        'eng':[timestamp, engagement_value]
    """

    pkl_folder = "./UE-HRI/Pickles/"

    def __init__(self, name):
        self.name = name
        self.fn = f"Data_{self.name}.pkl"
        self.stats = pd.DataFrame(columns=["ts","num_faces"])
        self.prep = []
        self.update({"name": self.name})

    def to_pkl(self):
        # check if bag is there
        bag = f"UE-HRI/bags/{self.name}.bag"
        assert Path(bag).exists(), f"Cannot find {bag}!"
        self.bag = rosbag.Bag(bag)

        self.topics = list(self.bag.get_type_and_topic_info()[1].keys())
        topic_df = pd.DataFrame(self.topics)
        topic_df.to_csv("stats/topics_"+self.name)
        
        # check if elan file is there
        elan = f"UE-HRI/Annotation_ELAN/{self.name}.eaf"
        assert Path(elan).exists(), f"Cannot find in {elan}!"
        self.eaf = pympi.Elan.Eaf(elan)

        print("Converting to .pkl")
        self.get_frames()  # OK
        self.get_annotations()
        try:
            cp = self.copy()

            os.makedirs(UEHRI_DATA.pkl_folder, exist_ok=True)
            print(f"Writing {self.fn} to disk...")
            dill.dump(cp, open(UEHRI_DATA.pkl_folder + self.fn, "wb"))
            # pickle.dump(self.copy(), open(f'Data_{self.name}.pkl', 'wb'))
        except Exception as e:
            print(e)
            print(f"Failed to save {self.fn}")
        self.cleanup()

    def from_pkl(self):
        print("Reading from pickle file")
        try:
            # open a file, where you stored the pickled data
            file = open(UEHRI_DATA.pkl_folder + self.fn, "rb")

            # dump information to that file
            d = pickle.load(file)
            self.update(d)

            # convert the bytes image arrays to cv2
            self.img_to_cv2()

            # close the file
            file.close()
        except Exception as e:
            print(e)

    def img_to_cv2(self):
        front_frames_cv2 = []
        for fr in self.get("front"):
            image = np.frombuffer(fr["image"], dtype=np.uint8)
            front_frames_cv2.append({"ts": fr["ts"], "image": image})

        bottom_frames_cv2 = []
        for fr in self.get("bottom"):
            image = np.frombuffer(fr["image"], dtype=np.uint8)
            bottom_frames_cv2.append({"ts": fr["ts"], "image": image})

        self.update({"front": front_frames_cv2})
        self.update({"bottom": bottom_frames_cv2})

    def get_frames(self):
        front_frames = []
        bottom_frames = []
        
        init = 0
        for topic, msg, t in self.bag.read_messages():
            # print(topic)
            if topic == "/naoqi_driver_node/camera/front/image_raw":
                if init == 0:
                    init = t.to_sec()
                store_time = t.to_sec() - init
                img = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
                self.prep.append([t, self.get_num_faces(img)])
                front_frames.append({"ts": store_time, "image": img.tobytes()})
            if topic == "/naoqi_driver_node/camera/bottom/image_raw":
                if init == 0:
                    init = t.to_sec()
                store_time = t.to_sec() - init
                img = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
                bottom_frames.append({"ts": store_time, "image": img.tobytes()})

        self.update({"front": front_frames})
        self.update({"bottom": bottom_frames})
        print("Frames loaded")
        

    @staticmethod
    def get_num_faces(img):
        detector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx",
            "",
            (640, 480)
            )
        # If input is an image
        tm = cv.TickMeter()
        detector.setInputSize((img.shape[1], img.shape[0]))
        faces = detector.detect(img)
        if faces[1] is not None:
            # print(f"Number of faces: {len(faces[1])}")
            num_faces = len(faces[1])
            return num_faces
        return 0
            
            # visualize(img, faces, tm.getFPS())
            # cv.imshow("image1", faces[1])
        # print(faces)


            
    def get_annotations(self):
        self.annotations = {}
        for ort_tier in self.eaf.get_tier_names():
            self.annotations[ort_tier] = self.eaf.get_annotation_data_for_tier(ort_tier)
        print(f"Annotations loaded")

        self.engagement = []
        # getting engagement annotation
        for img in self.get("front"):
            for ann in self.annotations["Engagement"]:
                if img["ts"] >= ann[0] / 1000 and img["ts"] <= ann[1] / 1000:
                    self.engagement.append({img["ts"]: str(ann[2])})
                    break
                else:
                    self.engagement.append({img["ts"]: str()})
        self.update({"eng": self.engagement})
        print(f"Engagement loaded")

    def cleanup(self):
        self.bag.close()
        del self.bag
        print(f"Closed bag {self.get('name')}")

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
            cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if __name__ == "__main__":
    # folder structure
    # ./UE-HRI/bags/*.bag
    # ./UE-HRI/Annotation_ELAN/*.eaf
    name = "user106_2017-03-08"

    data = UEHRI_DATA(name=name)

    print("done")
