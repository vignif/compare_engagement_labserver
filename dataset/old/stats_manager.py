# take the list of bags to analize
# download file from ftp
# call the script that runs the bag
# once that is done store the info in a file
# delete the local file and download the new one
# iterate until the file all.csv is complete

import os, glob
from pathlib import Path
import pandas as pd
import time
import cv2
from return_codes import Code
from get_stats import STATS_UEHRI
import ftplib


store_progress = "progress.csv"


class GETBAG:
    def __init__(self):
        path = "/share/UE-HRI"

        self.client = ftplib.FTP_TLS(timeout=999)
        self.client.connect("ftps.tsi.telecom-paristech.fr", 21)

        # enable TLS
        self.client.auth()
        self.client.prot_p()

        self.client.login("francesco.vigni@unina.it", "301829")

        self.client.cwd(path)
        print("connected to ftp server!")
        # print(self.client.retrlines('LIST'))

    def get_bag(self, filename):
        self.client.retrbinary(
            "RETR " + filename, open("UE-HRI/bags/" + filename, "wb").write
        )
        print(f"bag {filename} downloaded")
        self.client.quit()


def analize(bag_name):
    print(f"analysing bag {bag_name}")

    data = STATS_UEHRI(name=bag_name)

    if data.bag is None:
        # Download bag
        dl = GETBAG()
        dl.get_bag(bag_name + ".bag")
        print()
    data.__init__(bag_name)

    data.store_topics()

    return Code.SUCCESS


def empty_space():
    print("emptying space")
    dir = "UE-HRI/bags/"
    filelist = glob.glob(os.path.join(dir, "*.bag"))
    for f in filelist:
        os.remove(f)


def run_stats():
    # if all_1.csv exists, read from there. Else read from all.csv

    src = "all.csv"
    all_df = pd.read_csv(src, header=0)
    sort_df = all_df.sort_values("name")
    # print(sort_df.iloc[1,0])
    t0 = time.time()
    print(len(sort_df))
    for n, s in sort_df.iterrows():
        # print(n)
        # print(f"{s[0]} {str(s[1]).strip()}")
        eaf_name = str(s[0]).strip()
        bag_name = s[0].split(".eaf")[0]
        init = time.time()
        ret = analize(bag_name)
        ret = None
        sort_df.iloc[n, 1] = ret
        sort_df.to_csv(store_progress, index=None)
        print(f"analized {bag_name} in {time.time()-init:.3f} seconds")
        # print(s)
        empty_space()
    duration = time.time() - t0
    print(f"Analized all bags in {duration:.3f} seconds")
    print()


if __name__ == "__main__":
    run_stats()
    # analize('user106_2017-03-08')
