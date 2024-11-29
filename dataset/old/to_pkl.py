from uehri_datastruct import UEHRI_DATA


if __name__ == "__main__":
    # Data structure
    # 'front':[timestamp, image]
    # 'bottom':[timestamp, image]
    # 'eng':[timestamp, engagement_value]
    name = "user106_2017-03-08"

    data = UEHRI_DATA(name=name)

    data.to_pkl()

    print("done")
