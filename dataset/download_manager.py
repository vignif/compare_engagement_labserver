#!/usr/bin/env python3

import os
import ftplib
from progressbar import ProgressBar
from pathlib import Path
import rosbag

# FTP credentials and path
FTP_HOST = "ftps.tsi.telecom-paristech.fr"
FTP_USER = "francesco.vigni@unina.it"
FTP_PASS = "301829"
FTP_PATH = "/share/UE-HRI"

# Local paths
LOCAL_BAGS_DIR = "./bags"
PROGRESS_FILE = "progress.csv"


class DownloadManager:
    def __init__(self):
        self.client = ftplib.FTP_TLS(timeout=999)

    def connect(self):
        try:
            self.client.connect(FTP_HOST, 21)
            self.client.auth()
            self.client.prot_p()
            self.client.login(FTP_USER, FTP_PASS)
            self.client.cwd(FTP_PATH)

            print("Connected to FTP server!")

        except Exception as e:
            print("FTP connection failed:", e)

    def get_bag(self, filename):
        self.client.sendcmd("TYPE i")
        local_filepath = os.path.join(LOCAL_BAGS_DIR, filename)
        
        print(f"Downloading: {filename}")

        try:
            # Retrieve file size
            sz = self.client.size(f"{FTP_PATH}/{filename}")

            # Download file with progress bar
            self.pbar = ProgressBar(maxval=sz)
            self.pbar.start()

            with open(local_filepath, "wb") as file:
                def file_write(data):
                    file.write(data)
                    self.pbar.update(file.tell())

                self.client.retrbinary(f"RETR {filename}", file_write)
                self.pbar.finish()
                print(f"{filename} downloaded successfully.")

        except Exception as e:
            print(f"Failed to download {filename}: {e}")

    def disconnect(self):
        try:
            self.client.quit()
            print("Disconnected from FTP server.")

        except Exception as e:
            print("Error disconnecting from FTP server:", e)

    def is_bag_valid(self, filepath):
        try:
            bag = rosbag.Bag(filepath)
            bag.close()
            return True
        except Exception as e:
            print(f"Invalid bag file {filepath}: {e}")
            return False

def main():
    # Create DownloadManager instance
    d = DownloadManager()

    # Connect to FTP server
    d.connect()
    bags = [
            "user1_2017-03-03.bag",
            "user108_2017-03-15.bag",
            "user14_2017-06-14.bag",
            "user16_2017-01-23.bag",
            "user184_2017-02-03.bag",
            "user191_2017-03-16.bag",
            "user201_2017-02-03.bag",
            "user218_2017-02-09.bag",
            "user23_2017-01-20.bag",
            "user230_2017-01-25.bag",
            "user279_2017-04-26.bag",
            "user28_2017-01-26.bag",
            "user28_2017-01-31.bag",
            "user315_2017-02-10.bag",
            "user36_2017-03-13.bag",
            "user4_2017-02-17.bag",
            "user41_2017-02-07.bag",
            "user555_2017-04-14.bag",
            "user60_2017-02-20.bag",
            "user66_2017-05-12.bag",
            "user8_2017-01-31.bag",
            "user8_2017-03-23.bag",
            "user95_2017-02-24.bag",
            "user104_2017-06-20.bag",
            "user2_2017-01-19.bag",
            "user215_2017-02-03.bag",
            "user23_2017-03-10.bag",
            "user307_2017-04-13.bag",
            "user334_2017-04-13.bag",
            "user350_2017-04-13.bag",
            "user53_2017-01-31.bag",
            "user68_2017-01-26.bag"
            ]
    for bag in bags:
        local_filepath = os.path.join(LOCAL_BAGS_DIR, bag)
        
        # Check if the file already exists
        if os.path.exists(local_filepath):
            print(f"{bag} already exists. Checking validity...")
            
            # Check if the existing file is valid
            if d.is_bag_valid(local_filepath):
                print(f"{bag} is valid. Skipping download.")
                continue
            else:
                print(f"{bag} is invalid. Re-downloading...")
                os.remove(local_filepath)
        
        # Download the bag file
        d.get_bag(bag)

    # Disconnect from FTP server
    d.disconnect()


if __name__ == "__main__":
    main()

# docker run -it -v $(pwd)/bags:/app/bags ue-hri
