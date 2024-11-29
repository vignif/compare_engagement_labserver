
import os
import numpy as np
import rosbag
import cv2
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import pympi

class UserData:
    image_enc = 'rgb8'
    image_size = (224, 224)  # Adjusted for ResNet input size

    def __init__(self, name):
        self.name = name if not name.endswith('.bag') else name[:-4]
        self.load_bag_and_annotations()
        self.process_images()
        self.extract_engagement()

    def load_bag_and_annotations(self):
        """Load .bag and .eaf files."""
        try:
            self.bag = rosbag.Bag(f'../bags/{self.name}.bag')
            self.eaf = pympi.Elan.Eaf(f'../Annotation_ELAN/{self.name}.eaf')
            print(f'Loaded {self.name}.bag and {self.name}.eaf')
        except Exception as e:
            print(f'Error loading files for {self.name}: {e}')
            self.bag = None
            self.eaf = None

    def process_images(self):
        """Extract images from ROS bag file."""
        if self.bag is None:
            return

        self.images = []
        init_time = None
        for topic, msg, t in self.bag.read_messages(topics=['/naoqi_driver_node/camera/front/image_raw']):
            if init_time is None:
                init_time = t.to_sec()

            timestamp = t.to_sec() - init_time
            img = bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_enc)
            img_resized = cv2.resize(img, self.image_size)
            self.images.append({'ts': timestamp, 'image': img_resized})

        print(f'Processed {len(self.images)} images for {self.name}')
        self.bag.close()

    def extract_engagement(self):
        """Extract engagement annotations and align with images."""
        if self.eaf is None:
            return

        self.engagement = []
        engagement_tier = self.eaf.get_annotation_data_for_tier("Engagement")
        
        for img in self.images:
            eng_value = 0
            for start, end, annotation in engagement_tier:
                if start / 1000 <= img['ts'] <= end / 1000:
                    eng_value = float(annotation)  # Convert to float if necessary
                    break
            img['eng'] = eng_value
            self.engagement.append(eng_value)

        print("Extracted engagement annotations.")