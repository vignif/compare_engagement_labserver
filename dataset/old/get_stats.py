from pathlib import Path
import pandas as pd
import rosbag
import csv

class STATS_UEHRI:
    def __init__(self, name) -> None:
        self.name = name
        self.fn = f"Data_{self.name}.pkl"
        self.stats = pd.DataFrame(columns=["ts","num_faces"])
        self.prep = []
        bag = f"UE-HRI/bags/{self.name}.bag"

        if not Path(bag).exists():
            print("bag does not exist, downloading it..")
            self.bag = None
        else:
            self.bag = rosbag.Bag(bag)

    def store_topics(self):
        # check if bag is there

        self.topics = list(self.bag.get_type_and_topic_info()[1].keys())
        topic_df = pd.DataFrame(self.topics)
        topic_df.to_csv("stats/topic_names_"+self.name)

        with open(f"stats/topic_values_{self.name}.csv", 'w', encoding='utf-8') as f:

            writer = csv.writer(f, delimiter=",", skipinitialspace=True)
            writer.writerow(['msg_type', 'message_count', 'connections', 'frequency'])
            for tup in self.bag.get_type_and_topic_info()[1].values():
                writer.writerow(tup)
